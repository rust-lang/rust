// https://github.com/rust-lang/rust/issues/155127

#![feature(fn_delegation)]
#![allow(incomplete_features)]

struct S;

fn foo() {}

impl S {
    #[deprecated]
    //~^ ERROR `#[deprecated]` attribute cannot be used on delegations
    //~| WARN this was previously accepted by the compiler but is being phased out
    reuse foo;
}

fn main() {}
