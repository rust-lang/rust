//@aux-build:proc_macro_attr.rs
#![warn(clippy::double_must_use, clippy::must_use_unit)]

extern crate proc_macro_attr;

use proc_macro_attr::{add_must_use_to_async, dummy};

#[add_must_use_to_async]
async fn function() -> Result<(), ()> {
    Ok(())
}

#[add_must_use_to_async]
async fn unit_function() {}

#[add_must_use_to_async]
trait AsyncTrait {
    async fn method(&self) -> Result<(), ()>;
}

struct Struct;

#[add_must_use_to_async]
impl Struct {
    async fn method(&self) -> Result<(), ()> {
        Ok(())
    }
}

#[dummy]
#[must_use]
#[inline]
fn user_must_use() -> Result<(), ()> {
    //~^ double_must_use
    Ok(())
}

fn main() {}
