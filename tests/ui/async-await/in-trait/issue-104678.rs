//@ edition:2021
//@ check-pass

use std::future::Future;
pub trait Pool {
    type Conn;

    #[allow(async_fn_in_trait)]
    async fn async_callback<'a, F: FnOnce(&'a Self::Conn) -> Fut, Fut: Future<Output = ()>>(
        &'a self,
        callback: F,
    ) -> ();
}

pub struct PoolImpl;
pub struct ConnImpl;

impl Pool for PoolImpl {
    type Conn = ConnImpl;

    async fn async_callback<'a, F: FnOnce(&'a Self::Conn) -> Fut, Fut: Future<Output = ()>>(
        &'a self,
        _callback: F,
    ) -> () {
        todo!()
    }
}

fn main() {}
