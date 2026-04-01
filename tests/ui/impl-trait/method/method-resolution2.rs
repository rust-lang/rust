//! Check that the method call does not constrain the RPIT to `i32`, even though
//! `i32` is the only type that satisfies the RPIT's trait bounds.

//@ revisions: current next
//@[next] compile-flags: -Znext-solver
//@[current] check-pass

trait Trait {}

impl Trait for i32 {}

struct Bar<T>(T);

impl Bar<u32> {
    fn bar(self) {}
}

impl<T: Trait> Bar<T> {
    fn bar(self) {}
}

fn foo(x: bool) -> Bar<impl Trait> {
    if x {
        let x = foo(false);
        x.bar();
        //[next]~^ ERROR: multiple applicable items in scope
    }
    Bar(42_i32)
}

fn main() {}
