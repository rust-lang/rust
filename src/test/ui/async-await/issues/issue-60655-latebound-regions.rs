// Test that opaque `impl Trait` types are allowed to contain late-bound regions.

// build-pass (FIXME(62277): could be check-pass?)
// edition:2018

#![feature(type_alias_impl_trait)]

use std::future::Future;

pub type Func = impl Sized;

// Late bound region should be allowed to escape the function, since it's bound
// in the type.
fn null_function_ptr() -> Func {
    None::<for<'a> fn(&'a ())>
}

async fn async_nop(_: &u8) {}

pub type ServeFut = impl Future<Output=()>;

// Late bound regions occur in the generator witness type here.
fn serve() -> ServeFut {
    async move {
        let x = 5;
        async_nop(&x).await
    }
}

fn main() {}
