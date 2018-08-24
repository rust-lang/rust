#![feature(decl_macro)]

mod foo {
    pub trait T {
        fn f(&self) {}
    }
    impl T for () {}
}

mod bar {
    use foo::*;
    pub macro m() { ().f() }
    fn f() { ::baz::m!(); }
}

mod baz {
    pub macro m() { ().f() } //~ ERROR no method named `f` found for type `()` in the current scope
    fn f() { ::bar::m!(); }
}

fn main() {}
