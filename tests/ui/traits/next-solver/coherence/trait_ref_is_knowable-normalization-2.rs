//@ compile-flags: -Znext-solver
//@ check-pass

use std::future::{Future, IntoFuture};
use std::pin::Pin;

// We check that this does not overlap with the following impl from std:
//     impl<P> Future for Pin<P> where P: DerefMut, <P as Deref>::Target: Future { .. }
// This should fail because we know ` <&mut Value as Deref>::Target: Future` not to hold.
// For this to work we have to normalize in the `trait_ref_is_knowable` check as we
// otherwise add an ambiguous candidate here.
//
// See https://github.com/rust-lang/trait-system-refactor-initiative/issues/51
// for more details.
struct Value;
impl<'a> IntoFuture for Pin<&'a mut Value> {
    type Output = ();
    type IntoFuture = Pin<Box<dyn Future<Output = ()> + Send>>;

    fn into_future(self) -> Self::IntoFuture {
        todo!()
    }
}

fn main() {}
