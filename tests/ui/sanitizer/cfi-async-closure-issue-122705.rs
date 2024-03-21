// Verifies that using async closure works.
//
//@ needs-sanitizer-cfi
//@ compile-flags: -Clto -Cprefer-dynamic=off -Ctarget-feature=-crt-static -Zsanitizer=cfi -Copt-level=0 --edition=2021
//@ run-pass

#![feature(async_closure)]

#[inline(never)]
fn foo<T>(_: T) {}

fn main() {
    let a = async move |_: i32, _: i32| {};
    foo(a);
}
