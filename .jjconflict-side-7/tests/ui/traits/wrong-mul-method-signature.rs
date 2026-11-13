// This test is to make sure we don't just ICE if the trait
// method for an operator is not implemented properly.
// (In this case the mul method should take &f64 and not f64)
// See: #11450

use std::ops::Mul;

struct Vec1 {
    x: f64
}

// Expecting value in input signature
impl Mul<f64> for Vec1 {
    type Output = Vec1;

    fn mul(self, s: &f64) -> Vec1 {
    //~^ ERROR method `mul` has an incompatible type for trait
        Vec1 {
            x: self.x * *s
        }
    }
}

struct Vec2 {
    x: f64,
    y: f64
}

// Wrong type parameter ordering
impl Mul<Vec2> for Vec2 {
    type Output = f64;

    fn mul(self, s: f64) -> Vec2 {
    //~^ ERROR method `mul` has an incompatible type for trait
        Vec2 {
            x: self.x * s,
            y: self.y * s
        }
    }
}

struct Vec3 {
    x: f64,
    y: f64,
    z: f64
}

// Unexpected return type
impl Mul<f64> for Vec3 {
    type Output = i32;

    fn mul(self, s: f64) -> f64 {
    //~^ ERROR method `mul` has an incompatible type for trait
        s
    }
}

pub fn main() {
    // Check that the usage goes from the trait declaration:

    let x: Vec1 = Vec1 { x: 1.0 } * 2.0; // this is OK

    let x: Vec2 = Vec2 { x: 1.0, y: 2.0 } * 2.0; // trait had reversed order
    //~^ ERROR mismatched types
    //~| ERROR mismatched types

    let x: i32 = Vec3 { x: 1.0, y: 2.0, z: 3.0 } * 2.0;
}
