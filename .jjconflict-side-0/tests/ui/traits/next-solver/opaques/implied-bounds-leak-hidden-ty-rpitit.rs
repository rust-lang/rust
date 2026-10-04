//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

// Regression test for trait-system-refactor-initiative#159.  We need to make sure
// that computing the implied assumptions of `foo` does not look into the hidden
// type of `impl Sized`, as doing so adds a `T: 'static` implied bound which
// its caller does not have to prove.
//
// In this test the opaque is introduced via an RPITIT synthetic associated type
// in the signature and a `Projection(synthetic_assoc_ty, opaque_ty)` clause in the
// `ParamEnv`. We're initially fixing this bug by incorrectly marking opaque types
// as rigid. This test makes sure we also do so for opaque types in the `ParamEnv`.

use std::any::Any;

struct Outlives<T: 'static>(Option<T>);

trait Trait {
    fn foo<T>(x: T) -> (Box<dyn Any>, impl Sized) {
        //[next]~^ ERROR the parameter type `T` may not live long enough
        (Box::new(x), Outlives::<T>(None))
        //~^ ERROR the parameter type `T` may not live long enough
        //~| ERROR the parameter type `T` may not live long enough
        //~| ERROR the parameter type `T` may not live long enough
        //[current]~| ERROR the parameter type `T` may not live long enough
    }
}

impl Trait for i32 {}
impl Trait for u32 {
    fn foo<T>(x: T) -> (Box<dyn Any>, impl Sized) {
        //[next]~^ ERROR the parameter type `T` may not live long enough
        (Box::new(x), Outlives::<T>(None))
        //~^ ERROR the parameter type `T` may not live long enough
        //~| ERROR the parameter type `T` may not live long enough
        //~| ERROR the parameter type `T` may not live long enough
        //[current]~| ERROR the parameter type `T` may not live long enough
    }
}


fn main() {
    let any = <i32 as Trait>::foo(String::from("temporary").as_str()).0;
    println!("{}", any.downcast_ref::<&str>().unwrap());

    let any = <u32 as Trait>::foo(String::from("temporary").as_str()).0;
    println!("{}", any.downcast_ref::<&str>().unwrap());
}
