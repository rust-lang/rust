use std::error::Request;
use std::ptr;
use std::task::{ContextBuilder, Provider, RawWaker, RawWakerVTable, Waker};

#[test]
fn test_waker_getters() {
    let raw_waker = RawWaker::new(ptr::without_provenance_mut(42usize), &WAKER_VTABLE);
    let waker = unsafe { Waker::from_raw(raw_waker) };
    assert_eq!(waker.data() as usize, 42);
    assert!(ptr::eq(waker.vtable(), &WAKER_VTABLE));

    let waker2 = waker.clone();
    assert_eq!(waker2.data() as usize, 43);
    assert!(ptr::eq(waker2.vtable(), &WAKER_VTABLE));
}

// Test the `Request` API.
#[derive(Debug)]
struct SomeConcreteType {
    some_string: String,
}

impl Provider for SomeConcreteType {
    fn provide<'a>(&'a self, request: &mut Request<'a>) {
        request
            .provide_ref::<String>(&self.some_string)
            .provide_ref::<str>(&self.some_string)
            .provide_value_with::<String>(|| "bye".to_owned());
    }

    fn provide_mut<'a>(&'a mut self, request: &mut Request<'a>) {
        request.provide_mut::<String>(&mut self.some_string);
    }
}

#[test]
fn test_context_provider() {
    let obj = &mut SomeConcreteType { some_string: "hello".to_owned() };
    let builder = ContextBuilder::from_waker(Waker::noop()).provider(obj);
    let mut cx = builder.build();

    assert_eq!(cx.request_ref::<String>().unwrap(), "hello");
    assert_eq!(cx.request_value::<String>().unwrap(), "bye");
    assert_eq!(cx.request_value::<u8>(), None);

    cx.request_mut::<String>().unwrap().push_str(" world");

    assert_eq!(obj.some_string, "hello world");
}

static WAKER_VTABLE: RawWakerVTable = RawWakerVTable::new(
    |data| RawWaker::new(ptr::without_provenance_mut(data as usize + 1), &WAKER_VTABLE),
    |_| {},
    |_| {},
    |_| {},
);

// https://github.com/rust-lang/rust/issues/102012#issuecomment-1915282956
mod nop_waker {
    use core::future::{Future, ready};
    use core::pin::Pin;
    use core::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};

    const NOP_RAWWAKER: RawWaker = {
        fn nop(_: *const ()) {}
        const VTAB: RawWakerVTable = RawWakerVTable::new(|_| NOP_RAWWAKER, nop, nop, nop);
        RawWaker::new(&() as *const (), &VTAB)
    };

    const NOP_WAKER: &Waker = &unsafe { Waker::from_raw(NOP_RAWWAKER) };

    const NOP_CONTEXT: Context<'static> = Context::from_waker(NOP_WAKER);

    fn poll_once<T, F>(f: &mut F) -> Poll<T>
    where
        F: Future<Output = T> + ?Sized + Unpin,
    {
        let mut cx = NOP_CONTEXT;
        Pin::new(f).as_mut().poll(&mut cx)
    }

    #[test]
    fn test_const_waker() {
        assert_eq!(poll_once(&mut ready(1)), Poll::Ready(1));
    }
}
