// Regression test for https://github.com/rust-lang/rust/issues/165169

//@check-pass

#![feature(guard_patterns)]

fn main() {
    let x: !;
    match 1 {
        ((y if x)) => {}
        _ => {}
    }
}
