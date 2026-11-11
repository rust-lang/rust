//@ run-pass
#![feature(fn_traits, unboxed_closures)]

use std::marker::PhantomData;

// Test that we are able to infer a suitable kind for a "recursive"
// closure.  As far as I can tell, coding up a recursive closure
// requires the good ol' [Y Combinator].
//
// [Y Combinator]: https://en.wikipedia.org/wiki/Fixed-point_combinator#Y_combinator

struct YCombinator<F,A,R> {
    func: F,
    marker: PhantomData<(A,R)>,
}

impl<F,A,R> YCombinator<F,A,R> {
    fn new(f: F) -> YCombinator<F,A,R> {
        YCombinator { func: f, marker: PhantomData }
    }
}

impl<A,R,F : Fn(&dyn Fn(A) -> R, A) -> R> Fn<(A,)> for YCombinator<F,A,R> {
    extern "rust-call" fn call(&self, (arg,): (A,)) -> R {
        (self.func)(self, arg)
    }
}

impl<A,R,F : Fn(&dyn Fn(A) -> R, A) -> R> FnMut<(A,)> for YCombinator<F,A,R> {
    extern "rust-call" fn call_mut(&mut self, args: (A,)) -> R { self.call(args) }
}

impl<A,R,F : Fn(&dyn Fn(A) -> R, A) -> R> FnOnce<(A,)> for YCombinator<F,A,R> {
    type Output = R;
    extern "rust-call" fn call_once(self, args: (A,)) -> R { self.call(args) }
}

fn main() {
    let factorial = |recur: &dyn Fn(u32) -> u32, arg: u32| -> u32 {
        if arg == 0 {1} else {arg * recur(arg-1)}
    };
    let factorial: YCombinator<_,u32,u32> = YCombinator::new(factorial);
    let r = factorial(10);
    assert_eq!(3628800, r);
}
