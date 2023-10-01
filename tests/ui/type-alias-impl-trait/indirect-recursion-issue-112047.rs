// edition: 2021
// build-fail
//~^^ ERROR cycle detected when computing layout of

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
    }
}

fn main() {
    recur(());
}
