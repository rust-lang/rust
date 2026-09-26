use core::sync::SyncView;
use std::cell::Cell;
use std::future::Future;
use std::marker::PhantomPinned;
use std::pin::{Pin, pin};
use std::task::{Context, Poll, Waker};

const EMPTY: Vec<u8> = SyncView::new(Vec::new()).into_inner();
const PROJECTED: u32 = {
    let mut value = 41;
    let view = SyncView::from_mut(&mut value);
    *Pin::into_inner(Pin::new(view).as_pin_mut()) += 1;
    let view = SyncView::from_pin_mut(Pin::new(&mut value)).get_mut();
    *Pin::new(&*view).as_pin_ref().get_ref()
};

fn require_sync<T: Sync + ?Sized>(_: &T) {}

fn call_mut(mut f: impl FnMut() -> u32) -> u32 {
    f()
}

fn call_shared(f: &impl Fn() -> u32) -> u32 {
    f()
}

fn complete(future: impl Future<Output = u32>) -> u32 {
    let mut context = Context::from_waker(Waker::noop());
    match pin!(future).poll(&mut context) {
        Poll::Ready(value) => value,
        Poll::Pending => panic!("expected an immediately ready future"),
    }
}

async fn call_async_once(f: impl AsyncFnOnce() -> u32) -> u32 {
    f().await
}

struct Pinned {
    value: Cell<u32>,
    _pin: PhantomPinned,
}

#[test]
fn sync_view_const() {
    assert!(EMPTY.is_empty());
    assert_eq!(PROJECTED, 42);
}

#[test]
fn sync_view_access() {
    let mut value = std::sync::SyncView::new(Cell::new(1));
    require_sync(&value);
    value.as_mut().set(2);
    assert_eq!(value.into_inner().get(), 2);

    let mut values = [1, 2];
    let view = SyncView::from_mut(&mut values[..]);
    view.as_mut()[0] = 3;
    assert_eq!(view.as_ref(), &[3, 2]);
}

#[test]
fn sync_view_pinned_projection() {
    let mut pinned = pin!(Pinned { value: Cell::new(4), _pin: PhantomPinned });
    let address = std::ptr::from_ref(pinned.as_ref().get_ref());
    let view = SyncView::from_pin_mut(pinned.as_mut());
    require_sync(view.as_ref().get_ref());
    let inner = view.as_pin_mut();
    assert_eq!(std::ptr::from_ref(inner.as_ref().get_ref()), address);
    inner.value.set(5);
    assert_eq!(pinned.value.get(), 5);
}

#[test]
fn sync_view_forwarding() {
    let counter = Cell::new(0);
    let increment = SyncView::new(|| {
        counter.set(counter.get() + 1);
        counter.get()
    });
    require_sync(&increment);
    assert_eq!(call_mut(increment), 1);
    assert_eq!(call_shared(&SyncView::new(|| 7)), 7);

    let future = SyncView::new(async {
        std::future::ready(()).await;
        counter.get()
    });
    require_sync(&future);
    assert_eq!(complete(future), 1);
    assert_eq!(complete(call_async_once(SyncView::new(async || counter.get()))), 1);
}
