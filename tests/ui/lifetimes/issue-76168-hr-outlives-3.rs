//@ edition:2018

#![feature(unboxed_closures)]
use std::future::Future;

async fn wrapper<F>(f: F)
//~^ ERROR: expected a `FnOnce(&'a mut i32)` closure, found `i32`
//~| ERROR: expected a `FnOnce(&'a mut i32)` closure, found `i32`
//~| ERROR: expected a `FnOnce(&'a mut i32)` closure, found `i32`
where
F:,
for<'a> <i32 as FnOnce<(&'a mut i32,)>>::Output: Future<Output = ()> + 'a,
//~^ ERROR: expected a `FnOnce(&'a mut i32)` closure, found `i32`
//~| ERROR: expected a `FnOnce(&'a mut i32)` closure, found `i32`
//~| ERROR: expected a `FnOnce(&'a mut i32)` closure, found `i32`
{
    //~^ ERROR: expected a `FnOnce(&'a mut i32)` closure, found `i32`
    let mut i = 41;
    &mut i;
}

fn main() {}
