//@ compile-flags: --crate-type=lib
//@ check-pass

#![allow(useless_ptr_null_checks)]

const FOO: &usize = &42;

pub const _: () = assert!(!(FOO as *const usize).is_null());

pub const _: () = assert!(!(42 as *const usize).is_null());

pub const _: () = assert!((0 as *const usize).is_null());

pub const _: () = assert!(std::ptr::null::<usize>().is_null());

pub const _: () = assert!(!("foo" as *const str).is_null());
