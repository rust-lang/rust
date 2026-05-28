//! Ensure the unused_attributes lint fires for externally exported functions with `#[inline]`,
//! because `#[inline]` is ignored for such functions.

#![crate_type = "lib"]
#![feature(linkage)]
#![deny(unused_attributes)]

#[inline]
//~^ ERROR: `#[inline]` is ignored on externally exported functions
#[no_mangle]
fn no_mangle() {}

#[inline]
//~^ ERROR: `#[inline]` is ignored on externally exported functions
#[export_name = "export_name"]
fn export_name() {}

#[inline]
//~^ ERROR: `#[inline]` is ignored on externally exported functions
#[linkage = "external"]
fn external_linkage() {}

#[inline]
fn normal() {}

#[inline]
#[linkage = "internal"] // not exported
fn internal_linkage() {}
