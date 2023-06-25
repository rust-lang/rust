// ignore-tidy-linelength
// edition:2021
// revisions: classic next
//[next] compile-flags: -Ztrait-solver=next
//[next] check-pass
//[classic] known-bug: #112347
//[classic] build-fail
//[classic] failure-status: 101
//[classic] normalize-stderr-test "note: .*\n\n" -> ""
//[classic] normalize-stderr-test "thread 'rustc' panicked.*\n" -> ""
//[classic] normalize-stderr-test "(error: internal compiler error: [^:]+):\d+:\d+: " -> "$1:LL:CC: "
//[classic] rustc-env:RUST_BACKTRACE=0

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
