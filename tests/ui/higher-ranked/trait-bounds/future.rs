// ignore-tidy-linelength
//@ edition:2021
//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@ check-pass

#![feature(unboxed_closures)]

use std::future::Future;

trait Trait {
    fn func(&self, _: &str);
}

impl<T> Trait for T
where
    for<'a> T: Fn<(&'a str,)> + Send + Sync,
    for<'a> <T as FnOnce<(&'a str,)>>::Output: Future<Output = usize> + Send,
{
    fn func(&self, _: &str) {
        println!("hello!");
    }
}

async fn strlen(x: &str) -> usize {
    x.len()
}

fn main() {
    strlen.func("hi");
}
