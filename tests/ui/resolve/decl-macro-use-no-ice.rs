//@ edition: 2024
#![feature(decl_macro)]

// Regression test for issue <https://github.com/rust-lang/rust/issues/150711>
// The compiler previously ICE'd during identifier resolution
// involving `macro` items and `use` inside a public macro.


mod foo {
    macro f() {}

    pub macro m() {
        use f;  //~ ERROR `f` is private, and cannot be re-exported
        f!();   //~ ERROR macro import `f` is private
    }
}

fn main() {
    foo::m!();
}
