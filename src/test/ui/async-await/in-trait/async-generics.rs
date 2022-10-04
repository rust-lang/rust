// edition: 2021

#![feature(async_fn_in_trait)]
#![allow(incomplete_features)]

trait MyTrait<T, U> {
    async fn foo(&self) -> &(T, U);
}
//~^^ ERROR the parameter type `U` may not live long enough
//~| ERROR the parameter type `T` may not live long enough

impl<T, U> MyTrait<T, U> for (T, U) {
    async fn foo(&self) -> &(T, U) {
        self
    }
}

fn main() {}
