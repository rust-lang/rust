// build-pass (FIXME(62277): could be check-pass?)

// revisions: min_tait full_tait
#![feature(min_type_alias_impl_trait)]
#![cfg_attr(full_tait, feature(type_alias_impl_trait))]
//[full_tait]~^ WARN incomplete

use std::fmt::Debug;

fn main() {
    type Existential = impl Debug;

    fn f() -> Existential {}
    println!("{:?}", f());
}
