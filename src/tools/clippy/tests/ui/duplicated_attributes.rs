//@aux-build:proc_macro_attr.rs
#![warn(clippy::duplicated_attributes, clippy::duplicated_attributes)] //~ duplicated_attributes
#![feature(rustc_attrs)]
#![cfg(any(unix, windows))]
#![allow(dead_code)]
#![allow(dead_code)] //~ duplicated_attributes
#![cfg(any(unix, windows))] // Should not warn!

#[macro_use]
extern crate proc_macro_attr;

#[cfg(any(unix, windows, target_os = "linux"))]
#[allow(dead_code)]
#[allow(dead_code)] //~ duplicated_attributes
#[cfg(any(unix, windows, target_os = "linux"))] // Should not warn!
fn foo() {}

#[cfg(unix)]
#[cfg(windows)]
#[cfg(unix)] // cfgs are not handled
fn bar() {}

// No warning:
#[rustc_on_unimplemented(on(Self = "&str", label = "`a"), on(Self = "alloc::string::String", label = "a"))]
trait Abc {}

#[proc_macro_attr::duplicated_attr()] // Should not warn!
fn babar() {}

#[allow(missing_docs, reason = "library for internal use only")]
#[allow(exported_private_dependencies, reason = "library for internal use only")]
fn duplicate_reason() {}

#[allow(dead_code)]
#[allow(dead_code, unused_variables)] //~ duplicated_attributes
fn overlapping_lint_lists() {}

// https://github.com/rust-lang/rust-clippy/issues/13238
#[derive(proc_macro_attr::DerivedAttrs)]
#[attr(
    status(code = 400, description = "bad request"),
    status(code = 400, description = "also bad request")
)] // Should not warn!
enum ErrorResponse {
    A,
    B,
}

// Even though these two are identical, only the proc-macro knows
// if repeating its helper attribute is meaningful, so don't warn.
#[derive(proc_macro_attr::DerivedAttrs)]
#[attr(status(code = 400))]
#[attr(status(code = 400))] // Should not warn!
struct IdenticalInvocations {
    field: u8,
}

// Attributes expanded from an enabled cfg_attr are checked like regular ones.
#[allow(dead_code)]
#[cfg_attr(all(), allow(dead_code))] //~ duplicated_attributes
fn through_cfg_attr() {}

fn main() {}
