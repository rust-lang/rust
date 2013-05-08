type i32x4 = i32 ^ 4;

fn test(e: f32) {
    e as i32x4; //~ ERROR expected `i32` but found `f32`
}

fn main() {}
