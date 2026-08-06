//@ edition: 2021

#![feature(impl_trait_in_assoc_type)]

use core::future::Future;

trait Recur {
    type Recur: Future<Output = ()>;

    fn recur(self) -> Self::Recur;
}

async fn recur(t: impl Recur) {
    t.recur().await;
}

impl Recur for () {
    type Recur = impl Future<Output = ()>;

    fn recur(self) -> Self::Recur {
        async move { recur(self).await; }
        //~^ ERROR recursion in an async block requires boxing
    }
}

fn main() {
    recur(());
}
