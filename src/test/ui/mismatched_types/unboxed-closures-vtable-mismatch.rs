#![feature(unboxed_closures)]

use std::ops::FnMut;

fn to_fn_mut<A,F:FnMut<A>>(f: F) -> F { f }

fn call_it<F:FnMut(isize,isize)->isize>(y: isize, mut f: F) -> isize {
//~^ NOTE required by this bound in `call_it`
//~| NOTE required by a bound in `call_it`
    f(2, y)
}

pub fn main() {
    let f = to_fn_mut(|x: usize, y: isize| -> isize { (x as isize) + y });
    //~^ NOTE found signature of `fn(usize, isize) -> _`
    let z = call_it(3, f);
    //~^ ERROR type mismatch
    //~| NOTE expected signature of `fn(isize, isize) -> _`
    //~| NOTE required by a bound introduced by this call
    println!("{}", z);
}
