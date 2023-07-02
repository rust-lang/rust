//@aux-build:proc_macros.rs:proc-macro
#![feature(lint_reasons)]
#![deny(clippy::allow_attributes_without_reason)]
#![allow(unfulfilled_lint_expectations)]

extern crate proc_macros;
use proc_macros::{external, with_span};

// These should trigger the lint
#[allow(dead_code)]
#[allow(dead_code, deprecated)]
#[expect(dead_code)]
// These should be fine
#[allow(dead_code, reason = "This should be allowed")]
#[warn(dyn_drop, reason = "Warnings can also have reasons")]
#[warn(deref_nullptr)]
#[deny(deref_nullptr)]
#[forbid(deref_nullptr)]

fn main() {
    external! {
        #[allow(dead_code)]
        fn a() {}
    }
    with_span! {
        span
        #[allow(dead_code)]
        fn b() {}
    }
}

// Make sure this is not triggered on `?` desugaring

pub fn trigger_fp_option() -> Option<()> {
    Some(())?;
    None?;
    Some(())
}

pub fn trigger_fp_result() -> Result<(), &'static str> {
    Ok(())?;
    Err("asdf")?;
    Ok(())
}
