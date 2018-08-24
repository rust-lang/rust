struct vec2 { y: f32 }
struct vec3 { y: f32, z: f32 }

fn make(v: vec2) {
    let vec3 { y: _, z: _ } = v;
    //~^ ERROR mismatched types
    //~| expected type `vec2`
    //~| found type `vec3`
    //~| expected struct `vec2`, found struct `vec3`
}

fn main() { }
