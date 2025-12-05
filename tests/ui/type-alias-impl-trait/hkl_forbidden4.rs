//! This test used to ICE because, while an error was emitted,
//! we still tried to remap generic params used in the hidden type
//! to the ones of the opaque type definition.

//@ edition: 2021

#![feature(type_alias_impl_trait)]
use std::future::Future;

type FutNothing<'a> = impl 'a + Future<Output = ()>;

async fn operation(_: &mut ()) -> () {
    call(operation).await
    //~^ ERROR: concrete type differs from previous
}

#[define_opaque(FutNothing)]
async fn call<F>(_f: F)
//~^ ERROR item does not constrain `FutNothing::{opaque#0}`
where
    for<'any> F: FnMut(&'any mut ()) -> FutNothing<'any>,
{}

fn main() {}
