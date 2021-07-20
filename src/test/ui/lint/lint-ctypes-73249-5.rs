// revisions: min_tait full_tait
#![feature(min_type_alias_impl_trait)]
#![cfg_attr(full_tait, feature(type_alias_impl_trait))]
//[full_tait]~^ WARN incomplete
#![deny(improper_ctypes)]

pub trait Baz { }

impl Baz for u32 { }

type Qux = impl Baz;

fn assign() -> Qux { 3 }

#[repr(transparent)]
pub struct A {
    //[min_tait]~^ ERROR type aliases of `impl Trait` are not allowed as field types in structs
    x: Qux,
}

extern "C" {
    pub fn lint_me() -> A;
    //[full_tait]~^ ERROR `extern` block uses type `impl Baz`, which is not FFI-safe
}

fn main() {}
