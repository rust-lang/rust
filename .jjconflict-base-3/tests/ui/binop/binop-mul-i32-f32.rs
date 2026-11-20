fn foo(x: i32, y: f32) -> f32 {
    x * y //~ ERROR cannot multiply `i32` by `f32`
}

fn main() {}
