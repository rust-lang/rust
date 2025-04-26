//@ known-bug: #130935
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
