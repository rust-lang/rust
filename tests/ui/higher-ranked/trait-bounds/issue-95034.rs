// check-pass
// compile-flags: --edition=2021 --crate-type=lib

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
