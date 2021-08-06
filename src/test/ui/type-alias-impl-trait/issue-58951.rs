// check-pass

#![feature(type_alias_impl_trait)]

type A = impl Iterator;

fn def_a() -> A {
    0..1
}

pub fn use_a() {
    def_a().map(|x| x);
}

fn main() {}
