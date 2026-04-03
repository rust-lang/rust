// Regression test for #112623
// This used to ICE with "assertion failure: Size(0 bytes) / Size(8 bytes)"
// in rustc_const_eval. Now it correctly reports errors without crashing.

#![feature(const_trait_impl)]

#[const_trait]
//~^ ERROR cannot find attribute `const_trait` in this scope
trait Func<T> {
    type Output;

    fn call_once(self, arg: T) -> Self::Output;
}

struct Closure;

impl const Func<&usize> for Closure {
    //~^ ERROR const `impl` for trait `Func` which is not `const`
    type Output = usize;

    fn call_once(&'static self, arg: &usize) -> Self::Output {
        //~^ ERROR method `call_once` has an incompatible type for trait
        *arg
    }
}

enum Bug<T = [(); Closure.call_once(&0)]> {
    //~^ ERROR cannot call non-const method
    V(T),
}

fn main() {}
