// aux-build:unstable_generic_param.rs

extern crate unstable_generic_param;

use unstable_generic_param::Trait;

struct S;

impl Trait<usize> for S {
    fn foo() -> usize { 0 }
}

fn main() {
    let _ = S;
}
