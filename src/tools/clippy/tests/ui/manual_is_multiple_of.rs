//@aux-build: proc_macros.rs
#![warn(clippy::manual_is_multiple_of)]

fn main() {}

#[clippy::msrv = "1.87"]
fn f(a: u64, b: u64) {
    let _ = a % b == 0; //~ manual_is_multiple_of
    let _ = (a + 1) % (b + 1) == 0; //~ manual_is_multiple_of
    let _ = a % b != 0; //~ manual_is_multiple_of
    let _ = (a + 1) % (b + 1) != 0; //~ manual_is_multiple_of

    let _ = a % b > 0; //~ manual_is_multiple_of
    let _ = 0 < a % b; //~ manual_is_multiple_of

    proc_macros::external! {
        let a: u64 = 23424;
        let _ = a % 4096 == 0;
    }
}

#[clippy::msrv = "1.86"]
fn g(a: u64, b: u64) {
    let _ = a % b == 0;
}
