// aux-build:unstable_generic_param.rs

extern crate unstable_generic_param;

use unstable_generic_param::{Trait1, Trait2};

struct R;

impl Trait1 for S {
    fn foo() -> () { () } // ok
}

struct S;

impl Trait1<usize> for S { //~ ERROR use of unstable library feature 'unstable_default'
    fn foo() -> usize { 0 }
}

impl Trait2<usize> for S {
    fn foo() -> usize { 0 } // ok
}

fn main() {
    let _ = S;
}
