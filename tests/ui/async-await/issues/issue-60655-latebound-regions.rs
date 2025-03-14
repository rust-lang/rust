// Test that opaque `impl Trait` types are allowed to contain late-bound regions.

//@ check-pass
//@ edition:2018

#![feature(type_alias_impl_trait)]

use std::future::Future;

pub type Func = impl Sized;

// Late bound region should be allowed to escape the function, since it's bound
// in the type.
#[define_opaque(Func)]
fn null_function_ptr() -> Func {
    None::<for<'a> fn(&'a ())>
}

async fn async_nop(_: &u8) {}

pub type ServeFut = impl Future<Output = ()>;

// Late bound regions occur in the coroutine witness type here.
#[define_opaque(ServeFut)]
fn serve() -> ServeFut {
    async move {
        let x = 5;
        async_nop(&x).await
    }
}

fn main() {}
