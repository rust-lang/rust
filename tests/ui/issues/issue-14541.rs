struct Vec2 { y: f32 }
struct Vec3 { y: f32, z: f32 }

fn make(v: Vec2) {
    let Vec3 { y: _, z: _ } = v;
    //~^ ERROR mismatched types
    //~| expected `Vec2`, found `Vec3`
}

fn main() { }
