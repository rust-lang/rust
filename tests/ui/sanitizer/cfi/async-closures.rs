// Check various forms of dynamic closure calls

//@ edition: 2021
//@ revisions: cfi kcfi
// FIXME(#122848) Remove only-linux once OSX CFI binaries work
//@ only-linux
//@ ignore-backends: gcc
//@ [cfi] needs-sanitizer-cfi
//@ [kcfi] needs-sanitizer-kcfi
//@ [cfi] compile-flags: -Ccodegen-units=1 -Clto -Cprefer-dynamic=off
//@ [cfi] compile-flags: -Zunstable-options -Csanitize=cfi
//@ [kcfi] compile-flags: -Cpanic=abort -Zpanic-abort-tests -Cprefer-dynamic=off
//@ [kcfi] compile-flags: -Zunstable-options -Csanitize=kcfi
//@ compile-flags: -Ctarget-feature=-crt-static -Cunsafe-allow-abi-mismatch=sanitize
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
