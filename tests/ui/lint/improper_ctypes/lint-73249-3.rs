#![feature(type_alias_impl_trait)]
#![deny(improper_ctypes, improper_ctype_definitions)]

pub trait Baz {}

impl Baz for u32 {}

type Qux = impl Baz;

#[define_opaque(Qux)]
fn assign() -> Qux {
    3
}

#[repr(C)]
pub struct A { //~ ERROR: `repr(C)` type uses type `Qux`
    x: Qux,
}

extern "C" {
    pub fn lint_me() -> A; //~ ERROR: `extern` block uses type `A`
}

fn main() {}
