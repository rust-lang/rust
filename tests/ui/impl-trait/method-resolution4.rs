//! The recursive method call yields the opaque type. The
//! `next` method call then constrains the hidden type to `&mut _`
//! because `next` takes `&mut self`. We never resolve the inference
//! variable, but get a type mismatch when comparing `&mut _` with
//! `std::iter::Empty`.

//@ revisions: current next
//@[next] compile-flags: -Znext-solver
//@[current] check-pass

fn foo(b: bool) -> impl Iterator<Item = ()> {
    if b {
        foo(false).next().unwrap();
        //[next]~^ ERROR type annotations needed
    }
    std::iter::empty()
}

fn main() {}
