//! This test checks that we do not need to implement
//! members, whose `where Self: Sized` bounds don't hold for the impl type.

trait Foo {
    fn foo()
    where
        Self: Sized;
}

impl Foo for () {
    fn foo() {}
}

// Must not be allowed
impl Foo for i32 {}
//~^ ERROR: not all trait items implemented, missing: `foo`

impl Foo for dyn std::fmt::Debug {}

#[deny(dead_code)]
impl Foo for dyn std::fmt::Display {
    fn foo() {}
    //~^ ERROR this item cannot be used as its where bounds are not satisfied
}

struct Struct {
    i: i32,
    tail: [u8],
}

impl Foo for Struct {}

// Ensure we only allow known-unsized types to be skipped
trait Trait {
    fn foo(self)
    where
        Self: Sized;
}
impl<T: ?Sized> Trait for T {}
//~^ ERROR: not all trait items implemented, missing: `foo`

fn main() {}
