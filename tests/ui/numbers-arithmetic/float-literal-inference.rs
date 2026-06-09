//@ run-pass
struct S {
    z: f64
}

pub fn main() {
    let x: f32 = 4.0;
    println!("{}", x);
    let y: f64 = 64.0;
    println!("{}", y);
    let z = S { z: 1.0 };
    println!("{}", z.z);
}
