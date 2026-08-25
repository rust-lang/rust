//@ check-pass
//@ edition: 2021

use std::future::Future;
use std::pin::Pin;
use std::{marker::PhantomData, sync::Mutex};

type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

pub struct Scope<'scope, 'env: 'scope> {
    enqueued: Mutex<Vec<BoxFuture<'scope, ()>>>,
    phantom: PhantomData<&'env ()>,
}

impl<'scope, 'env: 'scope> Scope<'scope, 'env> {
    pub fn spawn(&'scope self, future: impl Future<Output = ()> + Send + 'scope) {
        self.enqueued.lock().unwrap().push(Box::pin(future));
    }
}

fn scope_with_closure<'env, B>(_body: B) -> BoxFuture<'env, ()>
where
    for<'scope> B: AsyncFnOnce(&'scope Scope<'scope, 'env>),
{
    todo!()
}

type ScopeRef<'scope, 'env> = &'scope Scope<'scope, 'env>;

async fn go<'a>(value: &'a i32) {
    let closure = async |scope: ScopeRef<'_, 'a>| {
        let _future1 = scope.spawn(async {
            // Make sure that `*value` is immutably borrowed with lifetime of
            // `'a` and not with the lifetime of the containing coroutine-closure.
            let _v = *value;
        });
    };
    scope_with_closure(closure).await;
}

fn main() {}
