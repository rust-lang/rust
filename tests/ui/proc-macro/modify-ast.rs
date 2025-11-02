//@ run-pass
//@ proc-macro: modify-ast.rs
//@ ignore-backends: gcc

extern crate modify_ast;

use modify_ast::*;

#[derive(Foo)]
pub struct MyStructc {
    #[cfg_attr(FALSE, foo)]
    _a: i32,
}

macro_rules! a {
    ($i:item) => ($i)
}

a! {
    #[assert1]
    pub fn foo() {}
}

fn main() {
    let _a = MyStructc { _a: 0 };
    foo();
}
