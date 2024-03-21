#![warn(clippy::duplicated_attributes)]
#![cfg(any(unix, windows))]
#![allow(dead_code)]
#![allow(dead_code)] //~ ERROR: duplicated attribute
#![cfg(any(unix, windows))]
//~^ ERROR: duplicated attribute
//~| ERROR: duplicated attribute

#[cfg(any(unix, windows, target_os = "linux"))]
#[allow(dead_code)]
#[allow(dead_code)] //~ ERROR: duplicated attribute
#[cfg(any(unix, windows, target_os = "linux"))]
//~^ ERROR: duplicated attribute
//~| ERROR: duplicated attribute
fn foo() {}

fn main() {}
