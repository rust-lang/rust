// edition:2021
#![crate_type = "lib"]
#![no_std]

use core::fmt;
use core::error; //~ ERROR use of unstable library feature 'error_in_core'

#[derive(Debug)]
struct MyCoreError(());

impl fmt::Display for MyCoreError {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        unimplemented!()
    }
}

impl error::Error for MyCoreError {}

fn main() {}
