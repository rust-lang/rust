//@ edition:2021
//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@ check-pass

#![feature(type_alias_impl_trait)]
#![feature(closure_lifetime_binder)]

use std::future::Future;
use std::pin::Pin;

trait AsyncFn<I, R>: FnMut(I) -> Self::Fut {
    type Fut: Future<Output = R>;
}

impl<F, I, R, Fut> AsyncFn<I, R> for F
where
    Fut: Future<Output = R>,
    F: FnMut(I) -> Fut,
{
    type Fut = Fut;
}

async fn call<C, R, F>(mut ctx: C, mut f: F) -> Result<R, ()>
where
    F: for<'a> AsyncFn<&'a mut C, Result<R, ()>>,
{
    loop {
        match f(&mut ctx).await {
            Ok(val) => return Ok(val),
            Err(_) => continue,
        }
    }
}

trait Cap<'a> {}
impl<T> Cap<'_> for T {}

fn check(ctx: &mut usize) {
    let mut inner = 0;

    // Ensure that normalization preserves an opaque nested inside a structural type, not only an
    // opaque which is itself the direct expansion of the free alias.
    type Ret<'a, 'b: 'a> =
        Pin<Box<impl Future<Output = Result<usize, ()>> + 'a + Cap<'b>>>;

    call(ctx, for<'a, 'b> |c: &'a mut &'b mut usize| -> Ret<'a, 'b> {
        inner += 1;
        Box::pin(async move {
            let _c = c;
            Ok(1usize)
        })
    });
}

fn main() {}
