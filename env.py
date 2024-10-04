import pygame

import gymnasium as gym
from gymnasium import Env
from gymnasium.spaces import Box, Discrete

import math
from typing import Optional
import numpy as np
import random as rd
import os

# helpers

def lerp(a, b, t):
    return a + (b - a) * t


def distance_point_to_point(pt1, pt2):
    return (((pt1.x - pt2.x)**2 + (pt1.y - pt2.y)**2)**0.5)


def distance_point_to_line(pt, a, b, c):
    if a == 0.0 and b != 0.0:
        return abs(b * pt.y + c) / abs(b)
    if b == 0.0 and a != 0.0:
        return abs(a * pt.x + c) / abs(a)
    
    return abs(a*pt.x + b * pt.y + c) / math.sqrt(a*a + b*b)
    

def rotate(origin, point, angle):
    qx = origin[0] + math.cos(angle) * (point[0] - origin[0]) - math.sin(angle) * (point[1] - origin[1])
    qy = origin[1] + math.sin(angle) * (point[0] - origin[0]) + math.cos(angle) * (point[1] - origin[1])
    return [qx, qy]


def rotate_rect(a, b, c, d, angle):
    center = [(a[0] + c[0]) / 2.0, (a[1] + c[1]) / 2.0]
    a = rotate(center, a, angle)
    b = rotate(center, b, angle)
    c = rotate(center, c, angle)
    d = rotate(center, d, angle)
    return a, b, c, d


def get_intersection(a, b, c, d):

    t_top = (d.x - c.x) * (a.y - c.y) - (d.y - c.y) * (a.x - c.x)
    u_top = (c.y - a.y) * (a.x - b.x) - (c.x - a.x) * (a.y - b.y)
    bottom = (d.y - c.y) * (b.x - a.x) - (d.x - c.x) * (b.y - a.y)

    if bottom != 0:
        t = t_top / bottom
        u = u_top / bottom

        if t >= 0.0 and t <= 1.0 and u >= 0.0 and u <= 1.0:
            return [lerp(a.x, b.x, t), lerp(a.y, b.y, t), t]

    return None


def poly_intersect(polygon, segment):
    poly_count = len(polygon)
    for i in range(poly_count):
        pp1 = Point(polygon[i][0], polygon[i][1])
        pp2 = Point(polygon[(i+1)%poly_count][0], polygon[(i+1)%poly_count][1])
        touch = get_intersection(pp1, pp2, segment.p1, segment.p2)
        if touch is not None:
            return True
    
    return False


def segment_mean(segment):
    return Point((segment.p1.x + segment.p2.x) / 2.0, (segment.p1.y + segment.p2.y) / 2.0)

# car

class Car:
    def __init__(self, pos_x, pos_y, angle_start):

        self.Pos = Point(pos_x, pos_y)
        self.Size = [5.5, 2.0] # [length, width]
        self.WheelBase = 0.65 * self.Size[0]

        self.AngleStart = angle_start
        self.Angle = angle_start

        self.Speed = 0.0
        self.SpeedMax = 300.0
        
        self.Acceleration = 0.0
        self.AccelerationMax = 10.7

        self.Steering = 0.0
        self.SteeringMax = 3.0

        self.Drag = 0.0
        self.DragFactor = self.AccelerationMax / (self.SpeedMax * self.SpeedMax)

        self.Diagonal = math.sqrt((self.Size[0] * self.Size[0]) + (self.Size[1] * self.Size[1])) / 2.0
        self.Alpha = math.degrees(math.atan2(self.Size[1], self.Size[0]))

        self.Damaged = False
        self.Distance = 0.0
        self.RewardDistance = 0.0

        self.SensorCount = 5
        self.SensorLength = 100.0
        self.SensorSpread = math.pi / 2.0

        self.Sensors = None
        self.Rays = None
        self.Vertexes = None


    def reset(self, pos_x, pos_y):
        self.Pos = Point(pos_x, pos_y)
        
        self.Speed = 0.0
        self.Acceleration = 0.0
        self.Angle = self.AngleStart
        self.Steering = 0.0
        self.Drag = 0.0
        self.Distance = 0.0
        self.RewardDistance = 0.0
        self.Damaged = False

        self.Sensors = None
        self.Rays = None
        self.Vertexes = None


    def step(self, accel, steer, dt, borders):
        if self.Damaged == False:
            
            self.Steering -= self.SteeringMax * steer
            self.Steering = min(self.Steering, self.SteeringMax)
            self.Steering = max(self.Steering, -self.SteeringMax)
            
            self.Acceleration += self.AccelerationMax * accel
            self.Acceleration = min(self.Acceleration , self.AccelerationMax)
            self.Acceleration = max(self.Acceleration , -self.AccelerationMax)

            # apply acceleration

            self.Speed += self.Acceleration * dt
            self.Drag = self.Speed * self.Speed * self.DragFactor
            self.Speed -= self.Drag * dt
            self.Speed = np.clip(self.Speed, 0.0, self.SpeedMax)

            if abs(self.Speed) < 0.1:
                self.Speed = 0.0

            # apply steering

            radCarAngle = math.radians(self.Angle)
            radSteerAngle = math.radians(self.Steering)

            cosCarAngle = math.cos(radCarAngle)
            sinCarAngle = math.sin(radCarAngle)

            frontWheel = [self.Pos.x + (self.WheelBase / 2.0) * cosCarAngle,
                          self.Pos.y + (self.WheelBase / 2.0) * sinCarAngle]
            
            backWheel = [self.Pos.x - (self.WheelBase / 2.0) * cosCarAngle,
                         self.Pos.y - (self.WheelBase / 2.0) * sinCarAngle]
            
            frontWheel[0] += self.Speed * dt * math.cos(radCarAngle + radSteerAngle)
            frontWheel[1] += self.Speed * dt * math.sin(radCarAngle + radSteerAngle)
            backWheel[0] += self.Speed * dt * cosCarAngle
            backWheel[1] += self.Speed * dt * sinCarAngle

            center = self.Pos

            self.Pos = Point((frontWheel[0] + backWheel[0]) / 2.0, (frontWheel[1] + backWheel[1]) / 2.0)

            distance = distance_point_to_point(self.Pos, center)
            self.Distance += distance
            self.RewardDistance += distance

            self.Angle = math.atan2(frontWheel[1] - backWheel[1], frontWheel[0] - backWheel[0])
            self.Angle = math.degrees(self.Angle)

            radCarAngle = math.radians(self.Angle)
            radCarAlpha = math.radians(self.Alpha)

            self.Vertexes = []
            self.Vertexes.append([self.Pos.x - math.cos(math.pi + radCarAngle + radCarAlpha) * self.Diagonal,
                                  self.Pos.y - math.sin(math.pi + radCarAngle + radCarAlpha) * self.Diagonal])
            self.Vertexes.append([self.Pos.x - math.cos(math.pi + radCarAngle - radCarAlpha) * self.Diagonal,
                                  self.Pos.y - math.sin(math.pi + radCarAngle - radCarAlpha) * self.Diagonal])
            self.Vertexes.append([self.Pos.x - math.cos(radCarAngle + radCarAlpha) * self.Diagonal,
                                  self.Pos.y - math.sin(radCarAngle + radCarAlpha) * self.Diagonal])
            self.Vertexes.append([self.Pos.x - math.cos(radCarAngle - radCarAlpha) * self.Diagonal,
                                  self.Pos.y - math.sin(radCarAngle - radCarAlpha) * self.Diagonal])

            # asses damage
            self.Damaged = False
            for border in borders:
                if poly_intersect(self.Vertexes, border):
                    self.Damaged = True
                    break

        # locate car sensors and rays
        self.Sensors = []
        for i in range(self.SensorCount):
            angle = math.radians(self.Angle) + lerp(self.SensorSpread / -2.0, self.SensorSpread / 2.0, i / (self.SensorCount - 1))
            start = self.Pos
            end = Point(self.Pos.x + math.cos(angle) * self.SensorLength, self.Pos.y + math.sin(angle) * self.SensorLength)
            self.Sensors.append([start, end])

        self.Rays = []
        for i in range(len(self.Sensors)):
            ray = self._get_ray(self.Sensors[i], borders)
            self.Rays.append(ray)


    def render(self, surf, trans, zoom):
        color =pygame.Color(0,0,0)
        if self.Damaged:
            color = pygame.Color(255,0,0)

        if self.Sensors is not None:
            for i in range(len(self.Sensors)):
                if self.Rays[i] is not None:
                    end = [self.Rays[i][0], self.Rays[i][1]]
                    p1 = [self.Sensors[i][0].x * zoom + trans[0], self.Sensors[i][0].y * zoom + trans[1]]
                    p2 = [end[0] * zoom + trans[0], end[1] * zoom + trans[1]]
                    pygame.draw.line(surf, (255,255,0), p1, p2, 1)

        if self.Vertexes is not None:
            poly = [(pt[0] * zoom + trans[0], pt[1] * zoom + trans[1]) for pt in self.Vertexes]
            pygame.draw.polygon(surf, color, poly)


    def _get_ray(self, sensor, borders):
        touches = []
        for border in borders:
            touch = get_intersection(sensor[0], sensor[1], border.p1, border.p2)
            if touch is not None:
                touches.append(touch)
                break

        if len(touches) > 0:
            min_offset = 9999.0
            min_touch = None
            for touch in touches:
                min_offset = min(min_offset, touch[2])
                min_touch = touch
            return touch
        
        return None


# track

class Track:
    def __init__(self, surface_width, surface_height):
        self.StartPoint = 0
        self.PointCount = 0
        self.Width = 14.0
        self.AvgSpeed = 0.0

        self.SrufaceCenter = [surface_width >> 1, surface_height >> 1]

        self.Points = None
        self.Segments = None
        self.Borders = None
        self.Tiles = None
        self.Checkpoints = None


    def step(self):
        terminated = False
        count = 0
        for tcp in self.Checkpoints:
            if tcp.checked:
                count += 1

        if count >= self.PointCount:
            terminated = True

        return terminated
    

    def reset(self):
        for tcp in self.Checkpoints:
            tcp.checked = False


    def render(self, surf,  trans, zoom):
        for tile in self.Tiles:
           tile.draw(surf=surf, zoom=zoom, trans=trans)

        for border in self.Borders:
            border.draw(surf=surf, zoom=zoom, trans=trans)

        for tcp in self.Checkpoints:
            tcp.draw(surf=surf, zoom=zoom, trans=trans, width=1, color=pygame.Color((255,0,0)))
        self.Checkpoints[self.StartPoint].draw(surf=surf, zoom=zoom, trans=trans, width=5)


    def load_trk(self, width, height, name):
        point_index = -1
        left = 0
        top = 0
        right = 0
        bottom = 0

        angle_start = 0.0

        self.StartPoint = 0
        self.PointCount = 0
        self.Points = []
        self.Segments = []
        self.Borders = []
        self.Tiles = []
        self.Checkpoints = []

        path = os.path.join('tracks', name)
        file = open(path, 'r')
        content = file.readlines()

        for line in content:
            if "TrackStartPoint" in line:
                parsed = line.split(":")
                self.StartPoint = int(parsed[1])
                continue
    
            if "TrackPointCount" in line:
                parsed = line.split(":")
                self.PointCount = int(parsed[1])
                continue

            if "TrackCarAngle" in line:
                parsed = line.split(":")
                angle_start = float(parsed[1])
                continue
            
            if "TrackPoints" in line:
                point_index = 0
                continue
            
            if point_index > -1:
                parsed = line.split(",")
                self.Points.append(Point(float(parsed[1]), float(parsed[0])))
                point_index += 1
                continue

        file.close()

        top = 9999.0
        left = 9999.0
        bottom = -9999.0
        right = -9999.0
        for pt in self.Points:
            top = min(top, pt.y)
            left = min(left, pt.x)
            bottom = max(bottom, pt.y)
            right = max(right, pt.x)

        w_center = self.SrufaceCenter
        t_center = [(left + right) / 2.0, (top + bottom) / 2.0]

        top += w_center[1] - t_center[1]
        left += w_center[0] - t_center[0]
        bottom += w_center[1] - t_center[1]
        right += w_center[0] - t_center[0]

        for i in range(self.PointCount):
            self.Points[i].x += w_center[0] - t_center[0]
            self.Points[i].y += w_center[1] - t_center[1]

        for i in range(self.PointCount - 1):
            self.Segments.append(Segment(self.Points[i], self.Points[i+1]))

        self.Segments.append(Segment(self.Points[self.PointCount-1], self.Points[0]))

        p1 = self.Segments[0].p1
        angle = math.pi / 2.0 - self.Segments[0].angle

        bp11 = Point(p1.x - self.Width * math.cos(angle), p1.y + self.Width * math.sin(angle))
        bp21 = Point(p1.x + self.Width * math.cos(angle), p1.y - self.Width * math.sin(angle))

        for i in range(1, len(self.Segments)):
            p2 = self.Segments[i].p1

            bp12 = Point(p2.x - self.Width * math.cos(angle), p2.y + self.Width * math.sin(angle))
            self.Borders.append(Segment(bp11, bp12))
            bp22 = Point(p2.x + self.Width * math.cos(angle), p2.y - self.Width * math.sin(angle))
            self.Borders.append(Segment(bp21, bp22))

            self.Checkpoints.append(Checkpoint(bp11, bp21))

            self.Tiles.append(Polygon([bp11, bp21, bp22, bp12]))

            bp11 = bp12
            bp21 = bp22
            angle = math.pi / 2.0 - self.Segments[i].angle

        bp11 = bp12
        bp21 = bp22

        p2 = self.Segments[0].p1

        bp12 = Point(p2.x - self.Width * math.cos(angle), p2.y + self.Width * math.sin(angle))
        self.Borders.append(Segment(bp11, bp12))
        bp22 = Point(p2.x + self.Width * math.cos(angle), p2.y - self.Width * math.sin(angle))
        self.Borders.append(Segment(bp21, bp22))

        self.Checkpoints.append(Checkpoint(bp11, bp21))

        self.Tiles.append(Polygon([bp11, bp21, bp22, bp12]))

        return angle_start


# environment

class CircuitEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(self, circuit_name, zoom, render_mode: Optional[str] = None):
        # action space:
        #       trust / brake [-1, 1] (-1.0 to 0.0 = brake) (0.0 to 1.0 trust)
        #       turn [-1, 1] (-1.0 to 0.0 = left) (0.0 to 1.0 = right)
        self.action_space = Box(low=-1.0, high=1.0, shape=(2,), dtype="float32")
        # observation space:
        #       lidar_1..5 [0 (touch), 1 (far))]
        self.observation_space = Box(low=0.0, high=1.0, shape=(5,), dtype=np.float32)

        self.state = None

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # define render components

        self.ScreenWidth = 1920
        self.ScreenHeight = 1200
        self.HeaderSurfaceWidth = self.ScreenWidth - 100 
        self.HeaderSurfaceHeight = 50
        self.TrackSurfaceWidth = self.ScreenWidth - 100
        self.TrackSurfaceHeight = self.ScreenHeight - 200
        self.TrackSurfaceCx = self.TrackSurfaceWidth / 2.0
        self.TrackSurfaceCy = self.TrackSurfaceHeight / 2.0
        self.FooterSurfaceWidth = self.ScreenWidth - 100 
        self.FooterSurfaceHeight = 50
        self.Zoom = zoom

        self.screen = None
        self.clock = None

        self.episodes = 0
        self.steps = 0
        self.score = 0.0
        self.reward = 0.0
        self.time_reward = 0.0
        self.space_reward = 0.0
        self.prev_reward = 0.0

        # define track

        self.track = Track(self.TrackSurfaceWidth, self.TrackSurfaceHeight)
        angle_start = self.track.load_trk(width=self.TrackSurfaceWidth, height=self.TrackSurfaceHeight, name=circuit_name)

        # define car

        self.car = Car(self.track.Points[self.track.StartPoint].x, self.track.Points[self.track.StartPoint].y, angle_start)


    def step(self, action):

        terminated = False
        step_reward = 0.0

        fps = self.clock.get_fps()
        if fps > 10.0:
            dt = 1.0 / self.clock.get_fps()
        else:
            dt = 1.0 / 60.0

        accel = np.clip(action[0], -1.0, 1.0)
        steer = np.clip(action[1], -1.0, 1.0)

        self.car.step(accel, steer, dt, self.track.Borders)

        state = []
        for i in range(len(self.car.Sensors)):
            start = Point(self.car.Sensors[i][0].x, self.car.Sensors[i][0].y)
            end = Point(self.car.Sensors[i][1].x, self.car.Sensors[i][1].y)
            if self.car.Rays[i] is not None:
                end = Point(self.car.Rays[i][0], self.car.Rays[i][1])
            dist = distance_point_to_point(start, end)
            state.append(dist / self.car.SensorLength)

        if self.car.Damaged:
            terminated = True
            step_reward = -100
        else:        
            terminated = self.track.step()

        if terminated == False:
            self.steps += 1

            if self.car.RewardDistance >= 10.0:
                self.car.RewardDistance = 0.0
                self.space_reward += 1.0
                self.reward += 1.0

            self.time_reward -= 0.1
            self.reward -= 0.1
            step_reward = self.reward - self.prev_reward
            self.prev_reward = self.reward

        if self.render_mode == "human":
            self.render()

        return state, step_reward, terminated, False, {}
    
    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()


    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        print('Called reset')

        self.state = [rd.random(), rd.random(), rd.random(), rd.random(), rd.random()]

        self.score = max(self.score, self.space_reward)

        self.episodes += 1        
        self.steps = 0

        self.reward = 0.0
        self.prev_reward = 0.0
        self.time_reward = 0.0
        self.space_reward = 0.0

        self.car.reset(self.track.Points[self.track.StartPoint].x, self.track.Points[self.track.StartPoint].y)
        self.track.reset()        

        if self.render_mode == "human":
            self.render()

        return self.state, {}
    

    def render(self):
        if self.render_mode == "human":
            return self._render_frame()
        

    def _render_frame(self):
        pygame.font.init()
        if self.screen is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((self.ScreenWidth, self.ScreenHeight))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        surf_track = pygame.Surface((self.TrackSurfaceWidth, self.TrackSurfaceHeight))
        surf_track.fill('green')
        surf_header = pygame.Surface((self.HeaderSurfaceWidth, self.HeaderSurfaceHeight))
        surf_header.fill('blue')
        surf_footer = pygame.Surface((self.FooterSurfaceWidth, self.FooterSurfaceHeight))
        surf_footer.fill('blue')

        scroll_x = self.car.Pos.x * self.Zoom
        scroll_y = self.car.Pos.y * self.Zoom
        trans = (self.TrackSurfaceCx - scroll_x, self.TrackSurfaceCy - scroll_y)

        self.track.render(surf=surf_track, trans=trans, zoom=self.Zoom)
        self.car.render(surf=surf_track, trans=trans, zoom=self.Zoom)
        self._render_indicators(surf=surf_track)
        self._render_header(surf=surf_header)
        self._render_footer(surf=surf_footer)

        if self.render_mode == "human":
            self.screen.blit(surf_header, (50, 50))
            self.screen.blit(surf_track, (50, 100))
            self.screen.blit(surf_footer, (50, self.ScreenHeight - 100))
            pygame.event.pump()
            pygame.display.update()
        

    def _render_header(self, surf):
        font = pygame.font.Font("C:\Windows\Fonts\Arial.ttf", 24)
        text = font.render("Reinforcement Learning - Proximal Policy Optimization", True, (221, 221, 221), 'blue')
        text_rect = text.get_rect()
        text_rect.center = (384, 25)
        surf.blit(text, text_rect)


    def _render_footer(self, surf):
        font = pygame.font.Font("C:\windows\Fonts\Arial.ttf", 12)
        
        text = font.render("Episodes: %04i" % self.episodes, True, (221, 221, 221), 'blue')
        text_rect = text.get_rect()
        text_rect.center = (128, 32)
        surf.blit(text, text_rect)

        text = font.render("Steps: %04i" % self.steps, True, (221, 221, 221), 'blue')
        text_rect = text.get_rect()
        text_rect.center = (256, 32)
        surf.blit(text, text_rect)

        text = font.render("Score: %04.1f" % self.score, True, (221, 221, 221), 'blue')
        text_rect = text.get_rect()
        text_rect.center = (384, 32)
        surf.blit(text, text_rect)

        text = font.render("Distance: %04.1f" % self.car.Distance, True, (221, 221, 221), 'blue')
        text_rect = text.get_rect()
        text_rect.center = (512, 32)
        surf.blit(text, text_rect)

        text = font.render("Reward: %04.1f (%04.1f) (%04i)" % (self.reward, self.time_reward, self.space_reward), True, (221, 221, 221), 'blue')
        text_rect = text.get_rect()
        text_rect.center = (640, 32)
        surf.blit(text, text_rect)

        self.clock.tick()
        fps = self.clock.get_fps()

        text = font.render("FPS: %3.1f" % fps, True, (221, 221, 221), 'blue')
        text_rect = text.get_rect()
        text_rect.center = (1664, 32)
        surf.blit(text, text_rect)


    def _render_indicators(self, surf):
        font = pygame.font.Font("C:\Windows\Fonts\Arial.ttf", 12)

        text = font.render("steer", True, (0, 0, 0), 'green')
        text_rect = text.get_rect()
        text_rect.center = (self.TrackSurfaceWidth / 2.0 - 96, self.TrackSurfaceHeight - 96)
        surf.blit(text, text_rect)

        text = font.render("speed", True, (0, 0, 0), 'green')
        text_rect = text.get_rect()
        text_rect.center = (self.TrackSurfaceWidth / 2.0 + 96, self.TrackSurfaceHeight - 96)
        surf.blit(text, text_rect)

        cx = self.TrackSurfaceWidth / 2.0 - 96.0
        pygame.draw.line(surf, 'red', [cx, self.TrackSurfaceHeight - 72.0], [cx, self.TrackSurfaceHeight - 8.0])
        angle = 64.0 * (self.car.Steering / self.car.SteeringMax)
        polygon = [(cx, self.TrackSurfaceHeight - 64.0), 
                   (cx + angle, self.TrackSurfaceHeight - 64.0),
                   (cx + angle, self.TrackSurfaceHeight - 16.0),
                   (cx, self.TrackSurfaceHeight - 16.0)]
        pygame.draw.polygon(surf, color='blue', points=polygon)

        cx = self.TrackSurfaceWidth / 2.0 + 96.0
        pygame.draw.line(surf, 'red', [cx, self.TrackSurfaceHeight - 72.0], [cx, self.TrackSurfaceHeight - 8.0])
        angle = 64.0 * (self.car.Speed / self.car.SpeedMax)
        polygon = [(cx, self.TrackSurfaceHeight - 64.0), 
                   (cx + angle, self.TrackSurfaceHeight - 64.0),
                   (cx + angle, self.TrackSurfaceHeight - 16.0),
                   (cx, self.TrackSurfaceHeight - 16.0)]
        pygame.draw.polygon(surf, color='blue', points=polygon)


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


    def draw(self, surf, radius=9, color=pygame.Color(0,0,255)):
        pygame.draw.circle(surf, color, [int(self.x), int(self.y)], radius)


class Segment:
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2
        self.angle = math.atan2(p2.y - p1.y, p2.x - p1.x)

    def draw(self, surf, zoom, trans, width=2, color=pygame.Color(255,255,255)):
        p1 = [int(self.p1.x * zoom + trans[0]), int(self.p1.y * zoom + trans[1])]
        p2 = [int(self.p2.x * zoom + trans[0]), int(self.p2.y * zoom + trans[1])]
        pygame.draw.line(surf, color, p1, p2, width)


class Polygon:
    def __init__(self, points):
        self.points = points


    def draw(self, surf, zoom, trans, color=pygame.Color((192,192,192))):
        pts = []
        for point in self.points:
            pts.append((point.x * zoom + trans[0], point.y * zoom + trans[1]))
        pygame.draw.polygon(surf, color, pts)


class Checkpoint:
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2
        self.a = p2.y - p1.y
        self.b = p1.x - p2.x
        self.c = p1.y * (p2.x - p1.x) - p1.x * (p2.y - p1.y)
        self.checked = False


    def reset(self):
        self.checked = False


    def check(self, car):
        dist = distance_point_to_line(car, self.a, self.b, self.c)
        if dist < 10.0 and self.checked == False:
            self.checked = True
            return True

        return False


    def draw(self, surf, zoom, trans, width=1, color=pygame.Color(255,255,255)):
        p1 = [int(self.p1.x * zoom + trans[0]), int(self.p1.y * zoom + trans[1])]
        p2 = [int(self.p2.x * zoom + trans[0]), int(self.p2.y * zoom + trans[1])]
        pygame.draw.line(surf, color, p1, p2, width)
