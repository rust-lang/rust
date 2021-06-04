// revisions: min_tait full_tait
#![feature(min_type_alias_impl_trait)]
#![cfg_attr(full_tait, feature(type_alias_impl_trait))]
//[full_tait]~^ WARN incomplete
#![deny(improper_ctypes)]

pub trait Baz { }

impl Baz for u32 { }

type Qux = impl Baz;

fn assign() -> Qux { 3 }

#[repr(C)]
pub struct A {
    x: Qux,
}

extern "C" {
    pub fn lint_me() -> A; //~ ERROR: uses type `impl Baz`
}

fn main() {}
