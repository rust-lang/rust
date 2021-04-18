// run-pass
// Checks that type param defaults are allowed after const params.
// revisions: full min
#![cfg_attr(full, feature(const_generics))]
#![feature(const_generics_defaults)]
#![allow(incomplete_features)]
#![allow(dead_code)]

struct FixedOutput<'a, const N: usize, T=u32> {
    out: &'a [T; N],
}

trait FixedOutputter {
    fn out(&self) -> FixedOutput<'_, 10>;
}

fn main() {}
