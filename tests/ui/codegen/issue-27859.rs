//@ run-pass

#[inline(never)]
fn foo(a: f32, b: f32) -> f32 {
    a % b
}

#[inline(never)]
fn bar(a: f32, b: f32) -> f32 {
    ((a as f64) % (b as f64)) as f32
}

fn main() {
    let unknown_float = std::env::args().len();
    println!("{}", foo(4.0, unknown_float as f32));
    println!("{}", foo(5.0, (unknown_float as f32) + 1.0));
    println!("{}", bar(6.0, (unknown_float as f32) + 2.0));
    println!("{}", bar(7.0, (unknown_float as f32) + 3.0));
}
