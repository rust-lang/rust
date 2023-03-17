// compile-flags: -Ztrait-solver=next

#![feature(fn_traits)]
#![feature(unboxed_closures)]
#![feature(tuple_trait)]

use std::marker::Tuple;
use std::ops::Fn;

fn foo<F: Fn<T>, T: Tuple>(f: Option<F>, t: T) {
    let y = (f.unwrap()).call(t);
}

fn main() {
    foo::<fn() -> str, _>(None, ());
    //~^ expected a `Fn<_>` closure, found `fn() -> str`
}
