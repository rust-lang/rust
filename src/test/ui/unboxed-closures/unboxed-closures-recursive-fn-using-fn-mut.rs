#![feature(core, fn_traits, unboxed_closures)]

use std::marker::PhantomData;

// An erroneous variant of `run-pass/unboxed_closures-infer-recursive-fn.rs`
// where we attempt to perform mutation in the recursive function. This fails to compile
// because it winds up requiring `FnMut` which enforces linearity.

struct YCombinator<F,A,R> {
    func: F,
    marker: PhantomData<(A,R)>,
}

impl<F,A,R> YCombinator<F,A,R> {
    fn new(f: F) -> YCombinator<F,A,R> {
        YCombinator { func: f, marker: PhantomData }
    }
}

impl<A,R,F : FnMut(&mut dyn FnMut(A) -> R, A) -> R> FnMut<(A,)> for YCombinator<F,A,R> {
    extern "rust-call" fn call_mut(&mut self, (arg,): (A,)) -> R {
        (self.func)(self, arg)
            //~^ ERROR cannot borrow `*self` as mutable more than once at a time
    }
}

impl<A,R,F : FnMut(&mut dyn FnMut(A) -> R, A) -> R> FnOnce<(A,)> for YCombinator<F,A,R> {
    type Output = R;
    extern "rust-call" fn call_once(mut self, args: (A,)) -> R {
        self.call_mut(args)
    }
}

fn main() {
    let mut counter = 0;
    let factorial = |recur: &mut dyn FnMut(u32) -> u32, arg: u32| -> u32 {
        counter += 1;
        if arg == 0 {1} else {arg * recur(arg-1)}
    };
    let mut factorial: YCombinator<_,u32,u32> = YCombinator::new(factorial);
    let mut r = factorial(10);
    assert_eq!(3628800, r);
}
