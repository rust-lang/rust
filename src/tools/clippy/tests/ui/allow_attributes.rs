//@aux-build:proc_macros.rs
//@aux-build:proc_macro_derive.rs
#![allow(unused)]
#![warn(clippy::allow_attributes)]
#![no_main]

extern crate proc_macros;
use proc_macros::{external, with_span};

// Using clippy::needless_borrow just as a placeholder, it isn't relevant.

// Should lint
#[allow(dead_code)]
struct T1;

struct T2; // Should not lint
#[deny(clippy::needless_borrow)] // Should not lint
struct T3;
#[warn(clippy::needless_borrow)] // Should not lint
struct T4;
// `panic = "unwind"` should always be true
#[cfg_attr(panic = "unwind", allow(dead_code))]
struct CfgT;

#[allow(clippy::allow_attributes, unused)]
struct Allowed;

#[expect(clippy::allow_attributes)]
#[allow(unused)]
struct Expected;

fn ignore_external() {
    external! {
        #[allow(clippy::needless_borrow)] // Should not lint
        fn a() {}
    }
}

fn ignore_proc_macro() {
    with_span! {
        span
        #[allow(clippy::needless_borrow)] // Should not lint
        fn a() {}
    }
}

fn ignore_inner_attr() {
    #![allow(unused)] // Should not lint
}

#[clippy::msrv = "1.81"]
fn msrv_1_81() {
    #[allow(unused)]
    let x = 1;
}

#[clippy::msrv = "1.80"]
fn msrv_1_80() {
    #[allow(unused)]
    let x = 1;
}

#[deny(clippy::allow_attributes)]
fn deny_allow_attributes() -> Option<u8> {
    let allow = None;
    allow?;
    Some(42)
}

// Edge case where the generated tokens spans match on #[repr(transparent)] which tricks the proc
// macro check
#[repr(transparent)]
#[derive(proc_macro_derive::AllowLintSameSpan)] // This macro generates tokens with the same span as the whole struct and repr
struct IgnoreDerived;
