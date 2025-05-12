//@ compile-flags: -Z unstable-options

#![feature(rustc_attrs)]

#[rustc_lint_query_instability]
//~^ ERROR attribute should be applied to a function
struct Foo;

impl Foo {
    #[rustc_lint_query_instability(a)]
    //~^ ERROR malformed `rustc_lint_query_instability`
    fn bar() {}
}

fn main() {}
