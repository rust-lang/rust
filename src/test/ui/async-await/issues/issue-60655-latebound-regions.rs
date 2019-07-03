// Test that existential types are allowed to contain late-bound regions.

// build-pass (FIXME(62277): could be check-pass?)
// edition:2018

#![feature(async_await, existential_type)]

use std::future::Future;

pub existential type Func: Sized;

// Late bound region should be allowed to escape the function, since it's bound
// in the type.
fn null_function_ptr() -> Func {
    None::<for<'a> fn(&'a ())>
}

async fn async_nop(_: &u8) {}

pub existential type ServeFut: Future<Output=()>;

// Late bound regions occur in the generator witness type here.
fn serve() -> ServeFut {
    async move {
        let x = 5;
        async_nop(&x).await
    }
}

fn main() {}
