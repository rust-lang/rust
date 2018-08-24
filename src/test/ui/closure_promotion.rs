// ignore-compare-mode-nll

#![allow(const_err)]

// nll successfully compiles this. It is a bug.
// See https://github.com/rust-lang/rust/issues/52384
fn main() {
    let x: &'static _ = &|| { let z = 3; z }; //~ ERROR does not live long enough
}
