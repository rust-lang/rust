// aux-build:modify-ast.rs

extern crate modify_ast;

use modify_ast::*;

#[derive(Foo)]
pub struct MyStructc {
    #[cfg_attr(my_cfg, foo)]
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
