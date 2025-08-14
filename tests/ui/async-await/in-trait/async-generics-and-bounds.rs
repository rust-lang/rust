//@ known-bug: #130935
//@ edition: 2021

use std::fmt::Debug;
use std::hash::Hash;

trait MyTrait<T, U> {
    async fn foo(&self) -> &(T, U) where T: Debug + Sized, U: Hash;
}

impl<T, U> MyTrait<T, U> for (T, U) {
    async fn foo(&self) -> &(T, U) {
        self
    }
}

fn main() {}
