//@ known-bug: rust-lang/rust#145217
//@compile-flags: -Zlint-mir
#![feature(super_let)]
fn main() {
    super let Some(1) = Some(2) else { return };
}
