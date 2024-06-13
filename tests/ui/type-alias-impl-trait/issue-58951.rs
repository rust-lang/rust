//@ check-pass

#![feature(type_alias_impl_trait)]

mod helper {
    pub type A = impl Iterator;

    pub fn def_a() -> A {
        0..1
    }
}

pub fn use_a() {
    helper::def_a().map(|x| x);
}

fn main() {}
