// check-pass
// edition: 2021

#![feature(async_fn_in_trait)]
#![feature(impl_trait_projections)]
#![allow(incomplete_features)]

use std::fmt::Debug;

trait MyTrait<'a, 'b, T> where Self: 'a, T: Debug + Sized + 'b {
    type MyAssoc;

    async fn foo(&'a self, key: &'b T) -> Self::MyAssoc;
}

impl<'a, 'b, T: Debug + Sized + 'b, U: 'a> MyTrait<'a, 'b, T> for U {
    type MyAssoc = (&'a U, &'b T);

    async fn foo(&'a self, key: &'b T) -> (&'a U, &'b T) {
        (self, key)
    }
}

fn main() {}
