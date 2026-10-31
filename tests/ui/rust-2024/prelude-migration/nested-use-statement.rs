//! This test used to ICE because the lint did not handle
//! nested use trees in the new HIR representation.

//@ check-pass

#[warn(rust_2024_prelude_collisions)]
fn main() {
    ().into_future()
    //~^ WARN: `into_future` will become ambiguous
    //~| WARN: is a hard error in Rust 2024
}
#[rustfmt::skip]
use future::{IntoFuture};
mod future {
    pub trait IntoFuture {
        fn into_future(&self) {}
    }
    impl IntoFuture for () {}
}
