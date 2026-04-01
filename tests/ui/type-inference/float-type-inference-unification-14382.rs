//! Regression test for https://github.com/rust-lang/rust/issues/14382

//@ run-pass
#[derive(Debug)]
struct Matrix4<S>(#[allow(dead_code)] S);
trait POrd<S> {}

fn translate<S: POrd<S>>(s: S) -> Matrix4<S> { Matrix4(s) }

impl POrd<f32> for f32 {}
impl POrd<f64> for f64 {}

fn main() {
    let x = 1.0;
    let m : Matrix4<f32> = translate(x);
    println!("m: {:?}", m);
}
