struct Vec2 { y: f32 }
struct Vec3 { y: f32, z: f32 }

fn make(v: Vec2) {
    let Vec3 { y: _, z: _ } = v;
    //~^ ERROR mismatched types
    //~| expected type `Vec2`
    //~| found type `Vec3`
    //~| expected struct `Vec2`, found struct `Vec3`
}

fn main() { }
