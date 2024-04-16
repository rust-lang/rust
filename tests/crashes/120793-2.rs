//@ known-bug: #120793
// can't use build-fail, because this also fails check-fail, but
// the ICE from #120787 only reproduces on build-fail.
//@ compile-flags: --emit=mir

#![feature(effects)]

trait Dim {
    fn dim() -> usize;
}

enum Dim3 {}

impl Dim for Dim3 {
    fn dim(x: impl Sized) -> usize {
        3
    }
}

fn main() {
    [0; Dim3::dim()];
}
