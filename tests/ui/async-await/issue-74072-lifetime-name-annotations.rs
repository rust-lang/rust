//@ edition:2018
use std::future::Future;

// test the quality of annotations giving lifetimes names (`'1`) when async constructs are involved

pub async fn async_fn(x: &mut i32) -> &i32 {
    let y = &*x;
    *x += 1; //~ ERROR cannot assign to `*x` because it is borrowed
    y
}

pub fn async_closure(x: &mut i32) -> impl Future<Output=&i32> {
    (async move || {
        //~^ ERROR lifetime may not live long enough
        //~| ERROR temporary value dropped while borrowed
        let y = &*x;
        *x += 1; //~ ERROR cannot assign to `*x` because it is borrowed
        y
    })()
}

pub fn async_closure_explicit_return_type(x: &mut i32) -> impl Future<Output=&i32> {
    (async move || -> &i32 {
        //~^ ERROR lifetime may not live long enough
        //~| ERROR temporary value dropped while borrowed
        let y = &*x;
        *x += 1; //~ ERROR cannot assign to `*x` because it is borrowed
        y
    })()
}

pub fn async_block(x: &mut i32) -> impl Future<Output=&i32> {
    async move {
        let y = &*x;
        *x += 1; //~ ERROR cannot assign to `*x` because it is borrowed
        y
    }
}

fn main() {}
