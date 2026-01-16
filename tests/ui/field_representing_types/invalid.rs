//@ revisions: old next
//@ [next] compile-flags: -Znext-solver
#![allow(incomplete_features)]
#![feature(field_projections)]

use std::field::field_of;

pub struct Struct {
    field: i32,
}

pub union Union {
    field: i32,
}

pub enum Enum {
    Variant { field: i32 },
}

pub trait Trait {}

fn main() {
    let _: field_of!(Struct); //~ ERROR: unexpected end of macro invocation
    let _: field_of!(Struct, field, extra); //~ ERROR: no rules expected `extra`
    // FIXME(FRTs): adjust error message to mention `field_of!`
    let _: field_of!(Enum, Variant..field); //~ ERROR: offset_of expects dot-separated field and variant names
    let _: field_of!(Struct, [42]); //~ ERROR: offset_of expects dot-separated field and variant names
}
