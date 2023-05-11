#![feature(negative_impls)]
#![deny(suspicious_auto_trait_impls)]

use std::marker::PhantomData;

struct ContainsVec<T>(Vec<T>);
impl !Send for ContainsVec<u32> {}
//~^ ERROR
//~| WARNING this will change its meaning

pub struct WithPhantomDataSend<T, U>(PhantomData<T>, U);
impl<T> !Send for WithPhantomDataSend<*const T, u8> {}
//~^ ERROR
//~| WARNING this will change its meaning

pub struct WithLifetime<'a, T>(&'a (), T);
impl<T> !Sync for WithLifetime<'static, Option<T>> {}
//~^ ERROR
//~| WARNING this will change its meaning

fn main() {}
