// run-rustfix
// ignore-test
//
// FIXME: Re-enable this test once we support choosing
// between multiple mutually exclusive suggestions for the same span

#![allow(warnings)]

struct A;
struct B;

fn foo() -> Result<A, B> {
    Ok(A)
}

fn bar() -> Result<A, B> {
    foo()?
    //~^ ERROR try expression alternatives have incompatible types [E0308]
}

fn main() {}
