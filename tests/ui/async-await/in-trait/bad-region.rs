//@ aux-build:bad-region.rs
//@ edition:2021

#![allow(async_fn_in_trait)]

extern crate bad_region as jewel;

use jewel::BleRadio;

pub struct Radio {}

impl BleRadio for Radio {
//~^ ERROR implicit elided lifetime not allowed here
    async fn transmit(&mut self) {}
}

fn main() {}
