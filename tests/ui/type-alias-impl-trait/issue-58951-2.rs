// check-pass

#![feature(type_alias_impl_trait)]

mod defining_use_scope {
    pub type A = impl Iterator;

    pub fn def_a() -> A {
        0..1
    }
}
use defining_use_scope::*;

pub fn use_a() {
    def_a().map(|x| x);
}

fn main() {}
