// revisions: min_tait full_tait
#![feature(min_type_alias_impl_trait)]
#![cfg_attr(full_tait, feature(type_alias_impl_trait))]
//[full_tait]~^ WARN incomplete
#![deny(improper_ctypes)]

pub trait Baz { }

impl Baz for u32 { }

type Qux = impl Baz;

pub trait Foo {
    type Assoc;
}

impl Foo for u32 {
    type Assoc = Qux;
}

fn assign() -> Qux { 1 }

extern "C" {
    pub fn lint_me() -> <u32 as Foo>::Assoc; //~ ERROR: uses type `impl Baz`
}

fn main() {}
