//@ edition: 2021
//@ check-pass

use std::future::Future;
use std::pin::Pin;

#[allow(async_fn_in_trait)]
pub trait MyTrait {
    async fn foo(&self) -> i32;
}

impl MyTrait for i32 {
    #[warn(refining_impl_trait)]
    fn foo(&self) -> Pin<Box<dyn Future<Output = i32> + '_>> {
        //~^ WARN impl trait in impl method signature does not match trait method signature
        Box::pin(async { *self })
    }
}

fn main() {}
