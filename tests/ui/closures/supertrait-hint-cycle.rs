//@ edition:2021
//@ check-pass

#![feature(type_alias_impl_trait)]
#![feature(closure_lifetime_binder)]

use std::future::Future;

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

fn works(ctx: &mut usize) {
    let mut inner = 0;

    type Ret<'a, 'b: 'a> = impl Future<Output = Result<usize, ()>> + 'a + Cap<'b>;

    let callback = for<'a, 'b> |c: &'a mut &'b mut usize| -> Ret<'a, 'b> {
        inner += 1;
        async move {
            let _c = c;
            Ok(1usize)
        }
    };
    call(ctx, callback);
}

fn doesnt_work_but_should(ctx: &mut usize) {
    let mut inner = 0;

    type Ret<'a, 'b: 'a> = impl Future<Output = Result<usize, ()>> + 'a + Cap<'b>;

    call(ctx, for<'a, 'b> |c: &'a mut &'b mut usize| -> Ret<'a, 'b> {
        inner += 1;
        async move {
            let _c = c;
            Ok(1usize)
        }
    });
}

fn main() {}
