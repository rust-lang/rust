#![warn(clippy::redundant_closure)]
#![no_std]

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;

fn issue_13895() {
    let _: Option<Vec<u8>> = true.then(|| vec![]);
    //~^ redundant_closure
}
