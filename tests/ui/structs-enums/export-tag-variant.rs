//@ run-pass
#![allow(non_camel_case_types)]

mod foo {
    pub enum t { t1, }
}

pub fn main() { let _v = foo::t::t1; }
