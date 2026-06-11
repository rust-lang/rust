//@ run-pass
#![feature(unboxed_closures)]
#![feature(fn_traits)]

struct Fun<F>(F);

impl<F, T> FnOnce<(T,)> for Fun<F> where F: Fn(T) -> T {
    type Output = T;

    extern "rust-call" fn call_once(self, (t,): (T,)) -> T {
        (self.0)(t)
    }
}

fn main() {
    let fun = Fun(|i: isize| i * 2);
    println!("{}", fun(3));
}
