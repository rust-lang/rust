// run-pass
// needs-unwind


#![feature(coroutines, coroutine_trait)]

use std::ops::Coroutine;
use std::pin::Pin;
use std::panic;

fn main() {
    let mut foo = || {
        if true {
            panic!();
        }
        yield;
    };

    let res = panic::catch_unwind(panic::AssertUnwindSafe(|| {
        Pin::new(&mut foo).resume(())
    }));
    assert!(res.is_err());

    for _ in 0..10 {
        let res = panic::catch_unwind(panic::AssertUnwindSafe(|| {
            Pin::new(&mut foo).resume(())
        }));
        assert!(res.is_err());
    }
}
