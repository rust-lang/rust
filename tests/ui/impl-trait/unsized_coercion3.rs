//! This test checks that opaque types get unsized instead of
//! constraining their hidden type to a trait object.

//@ revisions: next old
//@[next] compile-flags: -Znext-solver

trait Trait {}

impl Trait for u32 {}

fn hello() -> Box<impl Trait + ?Sized> {
    if true {
        let x = hello();
        //[next]~^ ERROR: the trait bound `dyn Send: Trait` is not satisfied
        let y: Box<dyn Send> = x;
        //[old]~^ ERROR: the size for values of type `impl Trait + ?Sized` cannot be know
    }
    Box::new(1u32)
}

fn main() {}
