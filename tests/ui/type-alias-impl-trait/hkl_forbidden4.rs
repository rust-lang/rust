//! This test used to ICE because, while an error was emitted,
//! we still tried to remap generic params used in the hidden type
//! to the ones of the opaque type definition.

//@ edition: 2021

#![feature(type_alias_impl_trait)]
use std::future::Future;

type FutNothing<'a> = impl 'a + Future<Output = ()>;
//~^ ERROR: unconstrained opaque type

async fn operation(_: &mut ()) -> () {
    //~^ ERROR: concrete type differs from previous
    call(operation).await
}

async fn call<F>(_f: F)
where
    for<'any> F: FnMut(&'any mut ()) -> FutNothing<'any>,
{
    //~^ ERROR: expected generic lifetime parameter, found `'any`
}

fn main() {}
