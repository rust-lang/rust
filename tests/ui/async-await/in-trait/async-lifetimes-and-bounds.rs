// check-pass
// edition: 2021

#![feature(async_fn_in_trait)]
#![allow(incomplete_features)]

use std::fmt::Debug;

trait MyTrait<'a, 'b, T> {
    async fn foo(&'a self, key: &'b T) -> (&'a Self, &'b T) where T: Debug + Sized;
}

impl<'a, 'b, T, U> MyTrait<'a, 'b, T> for U {
    async fn foo(&'a self, key: &'b T) -> (&'a U, &'b T) {
        (self, key)
    }
}

fn main() {}
