#![warn(clippy::duplicated_attributes)]
#![cfg(any(unix, windows))]
#![allow(dead_code)]
#![allow(dead_code)] //~ ERROR: duplicated attribute
#![cfg(any(unix, windows))] // Should not warn!

#[cfg(any(unix, windows, target_os = "linux"))]
#[allow(dead_code)]
#[allow(dead_code)] //~ ERROR: duplicated attribute
#[cfg(any(unix, windows, target_os = "linux"))] // Should not warn!
fn foo() {}

#[cfg(unix)]
#[cfg(windows)]
#[cfg(unix)] //~ ERROR: duplicated attribute
fn bar() {}

fn main() {}
