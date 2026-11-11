//@aux-build:proc_macro_attr.rs
//@no-rustfix
#![warn(clippy::double_must_use, clippy::must_use_unit)]
#![allow(dead_code)]

extern crate proc_macro_attr;

use proc_macro_attr::{add_must_use, add_must_use_to_async, dummy};

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

#[add_must_use]
struct MustUseStruct;

#[add_must_use]
union MustUseUnion {
    field: u32,
}

#[add_must_use]
enum MustUseEnum {
    Variant,
}

#[add_must_use]
trait MustUseTrait {}

impl MustUseTrait for u32 {}

#[add_must_use]
fn macro_must_use() -> Result<(), ()> {
    Ok(())
}

#[must_use]
fn returns_must_use_struct() -> MustUseStruct {
    //~^ double_must_use
    MustUseStruct
}

#[must_use]
fn returns_must_use_union() -> MustUseUnion {
    //~^ double_must_use
    MustUseUnion { field: 0 }
}

#[must_use]
fn returns_must_use_enum() -> MustUseEnum {
    //~^ double_must_use
    MustUseEnum::Variant
}

#[must_use]
fn returns_must_use_trait() -> impl MustUseTrait {
    //~^ double_must_use
    0u32
}

#[dummy]
#[must_use]
#[inline]
fn user_must_use() -> Result<(), ()> {
    //~^ double_must_use
    Ok(())
}

fn main() {}
