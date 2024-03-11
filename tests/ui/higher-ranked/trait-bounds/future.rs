// ignore-tidy-linelength
//@ edition:2021
//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@[next] check-pass
//@[current] known-bug: #112347
//@[current] build-fail
//@[current] failure-status: 101
//@[current] normalize-stderr-test "note: .*\n\n" -> ""
//@[current] normalize-stderr-test "thread 'rustc' panicked.*\n.*\n" -> ""
//@[current] normalize-stderr-test "(error: internal compiler error: [^:]+):\d+:\d+: " -> "$1:LL:CC: "
//@[current] rustc-env:RUST_BACKTRACE=0

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
