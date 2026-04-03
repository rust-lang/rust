//@ edition: 2021
// Regression test for #138710
// This used to ICE in rustc_middle::mir::interpret::queries
// when using min_generic_const_args with associated types and consts
// in an async context. Now it correctly reports errors without crashing.

#![feature(min_generic_const_args)]
//~^ WARN the feature `min_generic_const_args` is incomplete

#![allow(non_camel_case_types)]

trait B {
    type n: A;
}
trait A {
    const N: usize;
}
async fn fun(
) -> Box<dyn A> {
    //~^ ERROR the trait `A` is not dyn compatible
    *(&mut [0; <<Vec<u32> as B>::n as A>::N])
    //~^ ERROR the trait bound `Vec<u32>: B` is not satisfied
    //~| ERROR use of `const` in the type system not defined as `type const`
}
fn main() {}
