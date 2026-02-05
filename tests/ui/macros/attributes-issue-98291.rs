//! Regression test for #98291: attribute on macro_export + re-export should compile.
//! See also issue #150518 for a similar case.

//@ check-pass

#[rustfmt::skip]
#[macro_export]
macro_rules! _a {
    () => { "Hello world" };
}

pub use _a as a;

#[macro_export]
#[rust_analyzer::macro_style(braces)]
macro_rules! match_ast {
    (match $node:ident {}) => { $crate::match_ast!(match ($node) {}) };
    (match ($node:expr) {}) => {};
}

fn main() {
    println!(a!());

    match_ast! {
        match foo {}
    }
}
