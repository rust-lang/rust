//@ edition:2018

#![feature(decl_macro)]

mod foo {
    fn f() {}
    macro f() {}

    pub macro m() {
        use f as g; //~ ERROR `f` is private, and cannot be re-exported
        f!();
    }
}

fn main() {
    foo::m!();
}
