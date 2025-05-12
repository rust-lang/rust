#![feature(fn_traits)]

trait CallSingle<A, B> {
    fn call(&self, a: A) -> B where Self: Sized, Self: Fn(A) -> B;
}

impl<A, B, F: Fn(A) -> B> CallSingle<A, B> for F {
    fn call(&self, a: A) -> B {
        <Self as Fn(A) -> B>::call(self, (a,))
        //~^ ERROR associated item constraints are not allowed here
    }
}

fn main() {}
