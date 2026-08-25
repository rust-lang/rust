//! The mere presence of a match guard should not deem bound variables "used".
//! Regression test for https://github.com/rust-lang/rust/issues/151983
//@ check-pass
#![warn(unused)]
fn main() {
    match Some(42) {
        Some(unused) if true => (), //~WARN unused variable: `unused`
        _ => (),
    }
}
