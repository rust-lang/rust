//@ check-pass
//@ edition:2018

use std::future::Future;

async fn enter<'a, F, R>(mut callback: F)
where
    F: FnMut(&'a mut i32) -> R,
    R: Future<Output = ()> + 'a,
{
    unimplemented!()
}

fn main() {}
