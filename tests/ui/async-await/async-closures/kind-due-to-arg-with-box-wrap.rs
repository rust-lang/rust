//@ edition: 2024

// Regression test for <https://github.com/rust-lang/rust/issues/140292>.

struct Test;

impl Test {
    async fn an_async_fn(&mut self) {
        todo!()
    }

    pub async fn uses_takes_asyncfn(&mut self) {
        takes_asyncfn(Box::new(async || self.an_async_fn().await));
        //~^ ERROR expected a closure that implements the `AsyncFn` trait, but this closure only implements `AsyncFnMut`
    }
}

async fn takes_asyncfn(_: impl AsyncFn()) {
    todo!()
}

fn main() {}
