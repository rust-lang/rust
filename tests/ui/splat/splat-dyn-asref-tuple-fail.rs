//! Test that `#[splat]` on `&dyn AsRef<T>` where `T: Tuple` is an error.

#![allow(incomplete_features)]
#![feature(splat)]
#![feature(tuple_trait)]

fn dyn_asref_splat<T>(#[splat] _t: &dyn AsRef<T>) where T: std::marker::Tuple {}
//~^ ERROR cannot use splat attribute

fn main() {
    let s: String = "hello".to_owned();
    dyn_asref_splat::<String>(&s);
    //~^ ERROR `String` is not a tuple
}
