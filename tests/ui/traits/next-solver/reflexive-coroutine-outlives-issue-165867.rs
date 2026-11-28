//@ edition: 2024
//@ compile-flags: -Znext-solver=globally
//@ check-pass

#![crate_type = "lib"]
#![allow(dead_code)]

use std::future::Future;

pub struct Ctx;

pub trait InitializeFn<'a, T>: FnOnce(&'a Ctx) -> Self::InitializeFuture + Send + 'static {
    type InitializeFuture: Future<Output = T> + Send + 'a;
}

impl<'a, T, F, Fut> InitializeFn<'a, T> for F
where
    F: FnOnce(&'a Ctx) -> Fut + Send + 'static,
    Fut: Future<Output = T> + Send + 'a,
{
    type InitializeFuture = Fut;
}

pub trait Initialize: Sized {
    fn validate() -> impl Future<Output = Option<impl for<'a> InitializeFn<'a, Self>>> + Send;
}

pub trait InitializeExt: Initialize {
    fn validate_ext() -> impl Future<Output = Option<impl for<'a> InitializeFn<'a, Self>>> + Send {
        async {
            let make = Self::validate().await?;
            Some(async move |ctx: &Ctx| make(ctx).await)
        }
    }
}

impl<T: Initialize> InitializeExt for T {}

pub struct Level0;
impl Initialize for Level0 {
    async fn validate() -> Option<impl for<'a> InitializeFn<'a, Self>> {
        Some(async move |_: &Ctx| Level0)
    }
}

// Each level adds another layer of higher-ranked async closure obligations.
// Without the optimization, nested type-outlives constraints cause an
// exponential increase in compilation time and memory usage.
macro_rules! nest {
    ($outer:ident, $inner:ident) => {
        pub struct $outer($inner);
        impl Initialize for $outer {
            async fn validate() -> Option<impl for<'a> InitializeFn<'a, Self>> {
                let make = $inner::validate_ext().await?;
                Some(async move |ctx: &Ctx| $outer(make(ctx).await))
            }
        }
    };
}

nest!(Level1, Level0);
nest!(Level2, Level1);
nest!(Level3, Level2);
nest!(Level4, Level3);
nest!(Level5, Level4);
nest!(Level6, Level5);
nest!(Level7, Level6);
nest!(Level8, Level7);
nest!(Level9, Level8);

pub async fn top() -> Option<Level9> {
    Some(Level9::validate_ext().await?(&Ctx).await)
}
