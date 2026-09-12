//@ edition: 2021
//@ compile-flags: -Znext-solver

trait Trait {
    type Assoc<'a>
    where
        Self: 'a;
}

impl<T> Trait for T {
    type Assoc<'a> = ()
    where
        Self: 'a;
}

async fn inner<'a, T: Trait + 'a>(_: T, x: T::Assoc<'a>) -> T::Assoc<'a> {
    std::future::ready(x).await
}

async fn outer<'a>() {
    let x = 1u32;
    inner(&x, ()).await;
}

fn is_send<T: Send>(_: T) {}

fn main() {
    is_send(outer())
    //~^ ERROR implementation of `Send` is not general enough
}
