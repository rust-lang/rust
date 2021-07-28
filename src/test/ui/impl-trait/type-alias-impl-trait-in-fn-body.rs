// build-pass (FIXME(62277): could be check-pass?)

#![feature(type_alias_impl_trait)]

use std::fmt::Debug;

fn main() {
    type Existential = impl Debug;

    fn f() -> Existential {}
    println!("{:?}", f());
}
