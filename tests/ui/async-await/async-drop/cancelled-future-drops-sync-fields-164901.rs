//@ run-pass
//@ edition: 2024

// Regression test for <https://github.com/rust-lang/rust/issues/164901>.
// Types with only a sync `Drop` impl nested inside another ADT were treated
// as needing async drop, so a cancelled future leaked them.

#![feature(async_drop)]
#![allow(incomplete_features)]

use std::future::Future;
use std::pin::pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::task::{Context, Waker};

static DROPS: AtomicUsize = AtomicUsize::new(0);

struct D;
impl Drop for D {
    fn drop(&mut self) {
        DROPS.fetch_add(1, Ordering::Relaxed);
    }
}

struct Wrapper<T>(T);

fn poll_once_then_drop(f: impl Future<Output = ()>) {
    let mut f = Box::pin(f);
    let _ = f.as_mut().poll(&mut Context::from_waker(Waker::noop()));
}

fn main() {
    poll_once_then_drop(async {
        let _d = D;
        std::future::pending::<()>().await
    });
    assert_eq!(DROPS.load(Ordering::Relaxed), 1);

    poll_once_then_drop(async {
        let _o = Some(D);
        std::future::pending::<()>().await
    });
    assert_eq!(DROPS.load(Ordering::Relaxed), 2);

    poll_once_then_drop(async {
        let _w = Wrapper(D);
        std::future::pending::<()>().await
    });
    assert_eq!(DROPS.load(Ordering::Relaxed), 3);

    poll_once_then_drop(async {
        let _p = pin!(D);
        std::future::pending::<()>().await
    });
    assert_eq!(DROPS.load(Ordering::Relaxed), 4);
}
