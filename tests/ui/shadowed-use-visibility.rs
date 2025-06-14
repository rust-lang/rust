//@ run-pass

#![allow(unused_imports)]
mod foo {
    pub fn f() {}

    pub use self::f as bar;
    use crate::foo as bar;
}

fn main() {
    foo::bar();
}
