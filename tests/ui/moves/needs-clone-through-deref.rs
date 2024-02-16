//@ run-rustfix
#![allow(dead_code, noop_method_call)]
use std::ops::Deref;
struct S(Vec<usize>);
impl Deref for S {
    type Target = Vec<usize>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl S {
    fn foo(&self) {
        // `self.clone()` returns `&S`, not `Vec`
        for _ in self.clone().into_iter() {} //~ ERROR cannot move out of dereference of `S`
    }
}
fn main() {}
