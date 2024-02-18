//@ check-pass
//@ edition: 2021

trait MyTrait<'a, 'b, T> {
    #[allow(async_fn_in_trait)]
    async fn foo(&'a self, key: &'b T) -> (&'a Self, &'b T);
}

impl<'a, 'b, T, U> MyTrait<'a, 'b, T> for U {
    async fn foo(&'a self, key: &'b T) -> (&'a U, &'b T) {
        (self, key)
    }
}

fn main() {}
