//@ run-pass
#![allow(non_camel_case_types)]

// We can export tags without exporting the variants to create a simple
// sort of ADT.


mod foo {
    pub enum t { t1, }

    pub fn f() -> t { return t::t1; }
}

pub fn main() { let _v: foo::t = foo::f(); }
