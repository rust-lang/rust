//@ no-prefer-dynamic
#![crate_type = "rlib"]
//@ compile-flags:-g

struct S1;

impl S1 {
    fn f(&mut self) { }
}


struct S2;

impl S2 {
    fn f(&mut self) { }
}
