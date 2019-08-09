// build-pass (FIXME(62277): could be check-pass?)
// edition:2018

#![feature(async_await)]

use std::future::Future;

#[allow(unused)]
async fn enter<'a, F, R>(mut callback: F)
where
    F: FnMut(&'a mut i32) -> R,
    R: Future<Output = ()> + 'a,
{
    unimplemented!()
}

fn main() {}
