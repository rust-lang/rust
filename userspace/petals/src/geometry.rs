#![no_std]
use alloc::string::ToString;
use core::default::Default;
extern crate alloc;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Point {
    pub x: i32,
    pub y: i32,
}

impl Point {
    pub const fn new(x: i32, y: i32) -> Self {
        Self { x, y }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Size {
    pub width: i32,
    pub height: i32,
}

impl Size {
    pub const fn new(width: i32, height: i32) -> Self {
        Self { width, height }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Rect {
    pub origin: Point,
    pub size: Size,
}

impl Rect {
    pub const fn new(x: i32, y: i32, width: i32, height: i32) -> Self {
        Self {
            origin: Point::new(x, y),
            size: Size::new(width, height),
        }
    }

    pub fn x(&self) -> i32 {
        self.origin.x
    }
    pub fn y(&self) -> i32 {
        self.origin.y
    }
    pub fn width(&self) -> i32 {
        self.size.width
    }
    pub fn height(&self) -> i32 {
        self.size.height
    }

    pub fn intersection(&self, other: &Rect) -> Option<Rect> {
        let x0 = self.x().max(other.x());
        let y0 = self.y().max(other.y());
        let x1 = (self.x() + self.width()).min(other.x() + other.width());
        let y1 = (self.y() + self.height()).min(other.y() + other.height());

        if x1 > x0 && y1 > y0 {
            Some(Rect::new(x0, y0, x1 - x0, y1 - y0))
        } else {
            None
        }
    }

    pub const fn is_empty(&self) -> bool {
        self.size.width <= 0 || self.size.height <= 0
    }

    pub fn clip(self, bounds: Rect) -> Self {
        let x0 = self.x().max(bounds.x());
        let y0 = self.y().max(bounds.y());
        let x1 = (self.x() + self.width()).min(bounds.x() + bounds.width());
        let y1 = (self.y() + self.height()).min(bounds.y() + bounds.height());

        if x1 <= x0 || y1 <= y0 {
            Self::default()
        } else {
            Self::new(x0, y0, x1 - x0, y1 - y0)
        }
    }

    pub fn intersect(self, other: Rect) -> Self {
        self.intersection(&other).unwrap_or_default()
    }

    pub fn expand(self, px: i32) -> Self {
        Self::new(
            self.x() - px,
            self.y() - px,
            self.width() + px * 2,
            self.height() + px * 2,
        )
    }

    pub fn contains(&self, x: i32, y: i32) -> bool {
        x >= self.x()
            && x < self.x() + self.width()
            && y >= self.y()
            && y < self.y() + self.height()
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Color {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

impl Color {
    pub const fn new(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self { r, g, b, a }
    }

    pub const fn rgb(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b, a: 255 }
    }

    pub const fn to_u32(&self) -> u32 {
        ((self.a as u32) << 24) | ((self.r as u32) << 16) | ((self.g as u32) << 8) | (self.b as u32)
    }

    pub const fn from_u32(val: u32) -> Self {
        let a = ((val >> 24) & 0xFF) as u8;
        let r = ((val >> 16) & 0xFF) as u8;
        let g = ((val >> 8) & 0xFF) as u8;
        let b = (val & 0xFF) as u8;
        Self { r, g, b, a }
    }

    pub const BLACK: Self = Self::rgb(0, 0, 0);
    pub const WHITE: Self = Self::rgb(255, 255, 255);
    pub const TRANSPARENT: Self = Self::new(0, 0, 0, 0);
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Transform2D {
    pub a: f32,
    pub b: f32,
    pub c: f32,
    pub d: f32,
    pub tx: f32,
    pub ty: f32,
}

impl Default for Transform2D {
    fn default() -> Self {
        Self::identity()
    }
}

impl Transform2D {
    pub const fn identity() -> Self {
        Self {
            a: 1.0,
            b: 0.0,
            c: 0.0,
            d: 1.0,
            tx: 0.0,
            ty: 0.0,
        }
    }

    pub fn transform_point_f(&self, x: f32, y: f32) -> (f32, f32) {
        (
            self.a * x + self.c * y + self.tx,
            self.b * x + self.d * y + self.ty,
        )
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum EdgeAA {
    #[default]
    None,
    Coverage8,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct PointF {
    pub x: f32,
    pub y: f32,
}

impl PointF {
    pub const fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum PathVerb {
    MoveTo(PointF),
    LineTo(PointF),
    QuadTo(PointF, PointF),
    CubicTo(PointF, PointF, PointF),
    Close,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Path2D {
    pub verbs: alloc::vec::Vec<PathVerb>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum FillRule {
    NonZero,
    EvenOdd,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum LineCap {
    Butt,
    Round,
    Square,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum LineJoin {
    Miter,
    Round,
    Bevel,
}
