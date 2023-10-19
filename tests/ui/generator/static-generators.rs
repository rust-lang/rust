// run-pass

#![feature(coroutines, coroutine_trait)]

use std::pin::Pin;
use std::ops::{Coroutine, CoroutineState};

fn main() {
    let mut coroutine = static || {
        let a = true;
        let b = &a;
        yield;
        assert_eq!(b as *const _, &a as *const _);
    };
    // SAFETY: We shadow the original coroutine variable so have no safe API to
    // move it after this point.
    let mut coroutine = unsafe { Pin::new_unchecked(&mut coroutine) };
    assert_eq!(coroutine.as_mut().resume(()), CoroutineState::Yielded(()));
    assert_eq!(coroutine.as_mut().resume(()), CoroutineState::Complete(()));
}
