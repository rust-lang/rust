//@ revisions: e2021 e2024
//@[e2021] edition: 2021
//@[e2024] edition: 2024
//@ check-pass

#![deny(rust_2024_prelude_collisions)]

use core::future::IntoFuture;

struct Cat;

impl IntoFuture for Cat {
    type Output = ();
    type IntoFuture = core::future::Ready<()>;

    fn into_future(self) -> Self::IntoFuture {
        core::future::ready(())
    }
}

fn main() {
    let _ = Cat.into_future();
}
