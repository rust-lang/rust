//@ check-pass
//@ edition: 2021

use std::fmt::Debug;

trait MyTrait<'a, 'b, T> where Self: 'a, T: Debug + Sized + 'b {
    type MyAssoc;

    #[allow(async_fn_in_trait)]
    async fn foo(&'a self, key: &'b T) -> Self::MyAssoc;
}

impl<'a, 'b, T: Debug + Sized + 'b, U: 'a> MyTrait<'a, 'b, T> for U {
    type MyAssoc = (&'a U, &'b T);

    #[allow(async_fn_in_trait)]
    async fn foo(&'a self, key: &'b T) -> (&'a U, &'b T) {
        (self, key)
    }
}

fn main() {}
