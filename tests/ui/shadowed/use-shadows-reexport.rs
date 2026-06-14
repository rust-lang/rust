//! Check that a local `use` declaration can shadow a re-exported item within the same module.

//@ check-pass

#![allow(unused_imports)]
mod foo {
    pub fn f() {}

    pub use self::f as bar;
    use crate::foo as bar;
}

fn main() {
    foo::bar();
}
