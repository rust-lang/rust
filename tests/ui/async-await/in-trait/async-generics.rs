//@ check-fail
//@ known-bug: #102682
//@ edition: 2021

trait MyTrait<T, U> {
    async fn foo(&self) -> &(T, U);
}

impl<T, U> MyTrait<T, U> for (T, U) {
    async fn foo(&self) -> &(T, U) {
        self
    }
}

fn main() {}
