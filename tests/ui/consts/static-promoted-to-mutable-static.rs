//@ check-pass

#![allow(non_camel_case_types, non_upper_case_globals, static_mut_refs)]

pub struct wl_interface {
    pub version: i32,
}

pub struct Interface {
    pub other_interfaces: &'static [&'static Interface],
    pub c_ptr: Option<&'static wl_interface>,
}

pub static mut wl_callback_interface: wl_interface = wl_interface { version: 0 };

pub static WL_CALLBACK_INTERFACE: Interface =
    Interface { other_interfaces: &[], c_ptr: Some(unsafe { &wl_callback_interface }) };

// This static contains a promoted that points to a static that points to a mutable static.
pub static WL_SURFACE_INTERFACE: Interface =
    Interface { other_interfaces: &[&WL_CALLBACK_INTERFACE], c_ptr: None };

// And another variant of the same thing, this time with interior mutability.
use std::sync::OnceLock;
static LAZY_INIT: OnceLock<u32> = OnceLock::new();
static LAZY_INIT_REF: &[&OnceLock<u32>] = &[&LAZY_INIT];

fn main() {}
