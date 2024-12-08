#![feature(type_alias_impl_trait)]
#![deny(improper_ctypes)]

pub trait Baz {}

impl Baz for u32 {}

type Qux = impl Baz;

fn assign() -> Qux {
    3
}

#[repr(C)]
pub struct A {
    x: Qux,
}

extern "C" {
    pub fn lint_me() -> A; //~ ERROR: uses type `Qux`
}

fn main() {}
