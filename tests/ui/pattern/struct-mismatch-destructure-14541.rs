//! Regression test for https://github.com/rust-lang/rust/issues/14541

struct Vec2 { y: f32 }
struct Vec3 { y: f32, z: f32 }

fn make(v: Vec2) {
    let Vec3 { y: _, z: _ } = v;
    //~^ ERROR mismatched types
    //~| NOTE expected `Vec2`, found `Vec3`
    //~| NOTE this expression has type `Vec2`
}

fn main() { }
