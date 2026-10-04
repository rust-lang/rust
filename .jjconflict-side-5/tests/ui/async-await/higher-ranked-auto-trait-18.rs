// Repro for a bug where `PhantomData<*mut ()>` + `unsafe impl Send` fails to prove
// `Send` for an async block when the type parameter is a trait object (`dyn Trait`).
//
// The static assertion `assert_send::<Receiver<Box<dyn MyTrait>>>()` succeeds,
// but the same type captured across an `.await` in an async block does not.
// This is because MIR erases the `'static` lifetime from `dyn MyTrait + 'static`,
// and the auto-trait analysis cannot recover the `T: 'static` bound without
// higher-ranked assumptions.
//
// Using `PhantomData<Cell<()>>` instead of `PhantomData<*mut ()>` avoids the
// issue because `Cell<()>` is natively `Send`, so no `unsafe impl` (with its
// `T: 'static` bound) is needed.
//
// See <https://github.com/rust-lang/rust/issues/110338>.
//@ edition: 2021
//@ revisions: assumptions no_assumptions
//@[assumptions] compile-flags: -Zhigher-ranked-assumptions
//@[assumptions] check-pass
//@[no_assumptions] known-bug: #110338

use std::cell::Cell;
use std::future::Future;
use std::marker::PhantomData;
use std::pin::Pin;
use std::task::{Context, Poll};

// --- PhantomData<*mut ()> version: needs `unsafe impl Send` ---

struct Receiver<T: Send + 'static> {
    _value: Option<T>,
    _not_sync: PhantomData<*mut ()>,
}

unsafe impl<T: Send + 'static> Send for Receiver<T> {}

impl<T: Send + 'static> Future for Receiver<T> {
    type Output = T;
    fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
        Poll::Pending
    }
}

// --- PhantomData<Cell<()>> version: auto-derived Send works ---

struct ReceiverCell<T: Send + 'static> {
    _value: Option<T>,
    _not_sync: PhantomData<Cell<()>>,
}

impl<T: Send + 'static> Future for ReceiverCell<T> {
    type Output = T;
    fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
        Poll::Pending
    }
}

trait MyTrait: Send {}

fn require_send<F: Future + Send>(_f: F) {}

fn main() {
    // PhantomData<*mut ()> + concrete type: always works.
    require_send(async {
        let r = Receiver::<u32> { _value: None, _not_sync: PhantomData };
        let _ = r.await;
    });

    // PhantomData<*mut ()> + dyn Trait: fails without higher-ranked assumptions.
    require_send(async {
        let r = Receiver::<Box<dyn MyTrait>> { _value: None, _not_sync: PhantomData };
        let _ = r.await;
    });

    // PhantomData<Cell<()>> + dyn Trait: always works (auto-derived Send).
    require_send(async {
        let r = ReceiverCell::<Box<dyn MyTrait>> { _value: None, _not_sync: PhantomData };
        let _ = r.await;
    });
}
