// run-pass
// If `Mul` used an associated type for its output, this test would
// work more smoothly.

use std::ops::Mul;

#[derive(Copy, Clone)]
struct Vec2 {
    x: f64,
    y: f64
}

// methods we want to export as methods as well as operators
impl Vec2 {
#[inline(always)]
    fn vmul(self, other: f64) -> Vec2 {
        Vec2 { x: self.x * other, y: self.y * other }
    }
}

// Right-hand-side operator visitor pattern
trait RhsOfVec2Mul {
    type Result;

    fn mul_vec2_by(&self, lhs: &Vec2) -> Self::Result;
}

// Vec2's implementation of Mul "from the other side" using the above trait
impl<Res, Rhs: RhsOfVec2Mul<Result=Res>> Mul<Rhs> for Vec2 {
    type Output = Res;

    fn mul(self, rhs: Rhs) -> Res { rhs.mul_vec2_by(&self) }
}

// Implementation of 'f64 as right-hand-side of Vec2::Mul'
impl RhsOfVec2Mul for f64 {
    type Result = Vec2;

    fn mul_vec2_by(&self, lhs: &Vec2) -> Vec2 { lhs.vmul(*self) }
}

// Usage with failing inference
pub fn main() {
    let a = Vec2 { x: 3.0f64, y: 4.0f64 };

    // the following compiles and works properly
    let v1: Vec2 = a * 3.0f64;
    println!("{} {}", v1.x, v1.y);

    // the following compiles but v2 will not be Vec2 yet and
    // using it later will cause an error that the type of v2
    // must be known
    let v2 = a * 3.0f64;
    println!("{} {}", v2.x, v2.y); // error regarding v2's type
}
