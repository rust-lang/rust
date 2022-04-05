// edition:2021
#![crate_type = "lib"]
#![no_std]

use core::fmt;

#[derive(Debug)]
struct MyCoreError(());

impl fmt::Display for MyCoreError {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        unimplemented!()
    }
}

impl core::error::Error for MyCoreError {} //~ ERROR use of unstable library feature 'error_in_core'

fn main() {}
