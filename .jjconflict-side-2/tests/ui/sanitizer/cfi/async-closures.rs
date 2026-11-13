// Check various forms of dynamic closure calls

//@ edition: 2021
//@ revisions: cfi kcfi
// FIXME(#122848) Remove only-linux once OSX CFI binaries work
//@ only-linux
//@ ignore-backends: gcc
//@ [cfi] needs-sanitizer-cfi
//@ [kcfi] needs-sanitizer-kcfi
//@ compile-flags: -C target-feature=-crt-static -C unsafe-allow-abi-mismatch=sanitizer
//@ [cfi] compile-flags: -C codegen-units=1 -C lto -C prefer-dynamic=off -C opt-level=0
//@ [cfi] compile-flags: -Z sanitizer=cfi
//@ [kcfi] compile-flags: -Z sanitizer=kcfi
//@ [kcfi] compile-flags: -C panic=abort -Z panic-abort-tests -C prefer-dynamic=off
//@ run-pass

#![feature(async_fn_traits)]

use std::ops::AsyncFn;

#[inline(never)]
fn identity<T>(x: T) -> T { x }

// We can't actually create a `dyn AsyncFn()`, because it's dyn-incompatible, but we should check
// that we don't bug out when we encounter one.

fn main() {
   let f = identity(async || ());
   let _ = f.async_call(());
   let _ = f();
   let g: Box<dyn FnOnce() -> _> = Box::new(f) as _;
   let _ = g();
}
