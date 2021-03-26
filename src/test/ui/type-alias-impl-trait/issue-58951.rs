// check-pass

// revisions: min_tait full_tait
#![feature(min_type_alias_impl_trait)]
#![cfg_attr(full_tait, feature(type_alias_impl_trait))]
//[full_tait]~^ WARN incomplete

type A = impl Iterator;

fn def_a() -> A { 0..1 }

pub fn use_a() {
    def_a().map(|x| x);
}

fn main() {}
