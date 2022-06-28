// known-bug: #95034
// failure-status: 101
// compile-flags: --edition=2021 --crate-type=lib
// rustc-env:RUST_BACKTRACE=0

// normalize-stderr-test "thread 'rustc' panicked.*" -> "thread 'rustc' panicked"
// normalize-stderr-test "note:.*RUST_BACKTRACE=1.*\n" -> ""
// normalize-stderr-test "\nerror: internal compiler error.*\n\n" -> ""
// normalize-stderr-test "note:.*unexpectedly panicked.*\n\n" -> ""
// normalize-stderr-test "note: we would appreciate a bug report.*\n\n" -> ""
// normalize-stderr-test "note: compiler flags.*\n\n" -> ""
// normalize-stderr-test "note: rustc.*running on.*\n\n" -> ""
// normalize-stderr-test "query stack during panic:\n" -> ""
// normalize-stderr-test "we're just showing a limited slice of the query stack\n" -> ""
// normalize-stderr-test "end of query stack\n" -> ""
// normalize-stderr-test "#.*\n" -> ""

// This should not ICE.

// Refer to the issue for more minimized versions.

use std::{
    future::Future,
    marker::PhantomData,
    pin::Pin,
    task::{Context, Poll},
};

mod object {
    use super::*;

    pub trait Object<'a> {
        type Error;
        type Future: Future<Output = Self>;
        fn create() -> Self::Future;
    }

    impl<'a> Object<'a> for u8 {
        type Error = ();
        type Future = Pin<Box<dyn Future<Output = Self>>>;
        fn create() -> Self::Future {
            unimplemented!()
        }
    }

    impl<'a, E, A: Object<'a, Error = E>> Object<'a> for (A,) {
        type Error = ();
        type Future = CustomFut<'a, E, A>;
        fn create() -> Self::Future {
            unimplemented!()
        }
    }

    pub struct CustomFut<'f, E, A: Object<'f, Error = E>> {
        ph: PhantomData<(A::Future,)>,
    }

    impl<'f, E, A: Object<'f, Error = E>> Future for CustomFut<'f, E, A> {
        type Output = (A,);
        fn poll(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Self::Output> {
            unimplemented!()
        }
    }
}

mod async_fn {
    use super::*;

    pub trait AsyncFn {
        type Future: Future<Output = ()>;
        fn call(&self) -> Self::Future;
    }

    impl<F, Fut> AsyncFn for F
    where
        F: Fn() -> Fut,
        Fut: Future<Output = ()>,
    {
        type Future = Fut;
        fn call(&self) -> Self::Future {
            (self)()
        }
    }
}

pub async fn test() {
    use self::{async_fn::AsyncFn, object::Object};

    async fn create<T: Object<'static>>() {
        T::create().await;
    }

    async fn call_async_fn(inner: impl AsyncFn) {
        inner.call().await;
    }

    call_async_fn(create::<(u8,)>).await;
}
