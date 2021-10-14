// edition:2018
#![feature(unboxed_closures)]
use std::future::Future;

async fn wrapper<F>(f: F)
where for<'a> F: FnOnce<(&'a mut i32,)>,
    for<'a> <F as FnOnce<(&'a mut i32,)>>::Output: Future<Output=()> + 'a
{ //~ ERROR `<F as FnOnce<(&'a mut i32,)>>::Output` does not live long enough
    let mut i = 41;
    f(&mut i).await;
}

async fn add_one(i: &mut i32) {
    *i = *i + 1;
}

fn main() {}
