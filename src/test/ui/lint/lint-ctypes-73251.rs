// check-pass

// revisions: min_tait full_tait
#![feature(min_type_alias_impl_trait)]
#![cfg_attr(full_tait, feature(type_alias_impl_trait))]
//[full_tait]~^ WARN incomplete
#![deny(improper_ctypes)]

pub trait Foo {
    type Assoc;
}

impl Foo for () {
    type Assoc = u32;
}

type Bar = impl Foo<Assoc = u32>;

fn assign() -> Bar {}

extern "C" {
    pub fn lint_me() -> <Bar as Foo>::Assoc;
}

fn main() {}
