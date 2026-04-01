//@ run-pass
//@ proc-macro: attr-on-trait.rs
//@ ignore-backends: gcc

extern crate attr_on_trait;

use attr_on_trait::foo;

trait Foo {
    #[foo]
    fn foo() {}
}

impl Foo for i32 {
    fn foo(&self) {}
}

fn main() {
    3i32.foo();
}
