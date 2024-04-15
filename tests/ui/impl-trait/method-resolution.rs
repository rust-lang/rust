//! Check that we do not constrain hidden types during method resolution.
//! Otherwise we'd pick up that calling `bar` can be satisfied iff `u32`
//! is the hidden type of the RPIT.

//@ revisions: current next
//@[next] compile-flags: -Znext-solver
//@[next] check-pass

trait Trait {}

impl Trait for u32 {}

struct Bar<T>(T);

impl Bar<u32> {
    fn bar(self) {}
}

fn foo(x: bool) -> Bar<impl Sized> {
    //[current]~^ ERROR: cycle detected
    if x {
        let x = foo(false);
        x.bar();
        //[current]~^ ERROR: no method named `bar` found
    }
    todo!()
}

fn main() {}
