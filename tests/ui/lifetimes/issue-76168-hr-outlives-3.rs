//@ edition:2018

#![feature(unboxed_closures)]
use std::future::Future;

async fn wrapper<F>(f: F)
//~^ ERROR: expected `FnOnce(&'a mut i32)` signature, found `i32`
where
    F:,
    for<'a> <i32 as FnOnce<(&'a mut i32,)>>::Output: Future<Output = ()> + 'a,
{
    let mut i = 41;
    &mut i;
}

fn main() {}
