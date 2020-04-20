#![warn(clippy::unsafe_derive_deserialize)]
#![allow(unused, clippy::missing_safety_doc)]

extern crate serde;

use serde::Deserialize;

#[derive(Deserialize)]
pub struct A {}
impl A {
    pub unsafe fn new(_a: i32, _b: i32) -> Self {
        Self {}
    }
}

#[derive(Deserialize)]
pub struct B {}
impl B {
    pub unsafe fn unsafe_method(&self) {}
}

#[derive(Deserialize)]
pub struct C {}
impl C {
    pub fn unsafe_block(&self) {
        unsafe {}
    }
}

#[derive(Deserialize)]
pub struct D {}
impl D {
    pub fn inner_unsafe_fn(&self) {
        unsafe fn inner() {}
    }
}

// Does not derive `Deserialize`, should be ignored
pub struct E {}
impl E {
    pub unsafe fn new(_a: i32, _b: i32) -> Self {
        Self {}
    }

    pub unsafe fn unsafe_method(&self) {}

    pub fn unsafe_block(&self) {
        unsafe {}
    }

    pub fn inner_unsafe_fn(&self) {
        unsafe fn inner() {}
    }
}

// Does not have methods using `unsafe`, should be ignored
#[derive(Deserialize)]
pub struct F {}

fn main() {}
