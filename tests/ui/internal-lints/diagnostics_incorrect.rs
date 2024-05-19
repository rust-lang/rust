//@ compile-flags: -Z unstable-options

#![feature(rustc_attrs)]

#[rustc_lint_diagnostics]
//~^ ERROR attribute should be applied to a function
struct Foo;

impl Foo {
    #[rustc_lint_diagnostics(a)]
    //~^ ERROR malformed `rustc_lint_diagnostics`
    fn bar() {}
}

fn main() {}
