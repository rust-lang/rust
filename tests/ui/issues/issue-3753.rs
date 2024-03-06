//@ run-pass
// Issue #3656
// Issue Name: pub method preceded by attribute can't be parsed
// Abstract: Visibility parsing failed when compiler parsing

use std::f64;

#[derive(Copy, Clone)]
pub struct Point {
    x: f64,
    y: f64
}

#[derive(Copy, Clone)]
pub enum Shape {
    Circle(Point, f64),
    Rectangle(Point, Point)
}

impl Shape {
    pub fn area(&self, sh: Shape) -> f64 {
        match sh {
            Shape::Circle(_, size) => f64::consts::PI * size * size,
            Shape::Rectangle(Point {x, y}, Point {x: x2, y: y2}) => (x2 - x) * (y2 - y)
        }
    }
}

pub fn main(){
    let s = Shape::Circle(Point { x: 1.0, y: 2.0 }, 3.0);
    println!("{}", s.area(s));
}
