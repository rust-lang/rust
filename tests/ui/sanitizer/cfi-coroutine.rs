// Verifies that we can call dynamic coroutines

//@ revisions: cfi kcfi
// FIXME(#122848) Remove only-linux once OSX CFI binaries work
//@ only-linux
//@ [cfi] needs-sanitizer-cfi
//@ [kcfi] needs-sanitizer-kcfi
//@ compile-flags: -C target-feature=-crt-static
//@ [cfi] compile-flags: -C codegen-units=1 -C lto -C prefer-dynamic=off -C opt-level=0
//@ [cfi] compile-flags: -Z sanitizer=cfi
//@ [kcfi] compile-flags: -Z sanitizer=kcfi
//@ [kcfi] compile-flags: -C panic=abort -Z panic-abort-tests -C prefer-dynamic=off
//@ compile-flags: --test
//@ run-pass

#![feature(coroutines)]
#![feature(coroutine_trait)]

use std::ops::{Coroutine, CoroutineState};
use std::pin::{pin, Pin};

fn main() {
    let mut coro = |x: i32| {
        yield x;
        "done"
    };
    let mut abstract_coro: Pin<&mut dyn Coroutine<i32,Yield=i32,Return=&'static str>> = pin!(coro);
    assert_eq!(abstract_coro.as_mut().resume(2), CoroutineState::Yielded(2));
    assert_eq!(abstract_coro.as_mut().resume(0), CoroutineState::Complete("done"));
}
