// aux-build:unstable_generic_param.rs
#![feature(unstable_default6)]

extern crate unstable_generic_param;

use unstable_generic_param::*;

struct R;

impl Trait1 for S {
    fn foo() -> () { () } // ok
}

struct S;

impl Trait1<usize> for S { //~ ERROR use of unstable library feature 'unstable_default'
    fn foo() -> usize { 0 }
}

impl Trait1<isize> for S { //~ ERROR use of unstable library feature 'unstable_default'
    fn foo() -> isize { 0 }
}

impl Trait2<usize> for S { //~ ERROR use of unstable library feature 'unstable_default'
    fn foo() -> usize { 0 }
}

impl Trait3<usize> for S {
    fn foo() -> usize { 0 } // ok
}

fn main() {
}
