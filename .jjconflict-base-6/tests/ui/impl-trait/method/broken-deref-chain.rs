//@ revisions: current next
//@[next] compile-flags: -Znext-solver
//@[next] check-pass

// An annoying edge case of method selection. While computing the deref-chain
// constrains `T` to `u32`, the final method candidate does not and instead
// constrains to `i32`. In this case, we no longer check that the opaque
// remains unconstrained. Both method calls in this test constrain the opaque
// to `i32`.
use std::ops::Deref;

struct Foo<T, U>(T, U);
impl<U> Deref for Foo<u32, U> {
    type Target = U;
    fn deref(&self) -> &Self::Target {
        &self.1
    }
}

impl Foo<i32, i32> {
    fn method(&self) {}
}
fn inherent_method() -> impl Sized {
    if false {
        let x = Foo(Default::default(), inherent_method());
        x.method();
        let _: Foo<i32, _> = x; // Test that we did not apply the deref step
    }
    1i32
}

trait Trait {
    fn trait_method(&self) {}
}
impl Trait for Foo<i32, i32> {}
impl Trait for i32 {}
fn trait_method() -> impl Trait {
    if false {
        let x = Foo(Default::default(), trait_method());
        x.trait_method();
        let _: Foo<i32, _> = x; // Test that we did not apply the deref step
        //[current]~^ ERROR mismatched types
    }
    1i32
}

fn main() {}
