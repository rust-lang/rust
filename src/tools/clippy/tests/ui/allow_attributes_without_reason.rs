//@aux-build:proc_macros.rs
#![deny(clippy::allow_attributes_without_reason)]
#![allow(unfulfilled_lint_expectations, clippy::duplicated_attributes)]
//~^ allow_attributes_without_reason

extern crate proc_macros;
use proc_macros::{external, with_span};

// These should trigger the lint
#[allow(dead_code)]
//~^ allow_attributes_without_reason
#[allow(dead_code, deprecated)]
//~^ allow_attributes_without_reason
#[expect(dead_code)]
//~^ allow_attributes_without_reason
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

#[clippy::msrv = "1.81"]
fn msrv_1_81() {
    #[allow(unused)]
    //~^ allow_attributes_without_reason
    let _ = 1;
}

#[clippy::msrv = "1.80"]
fn msrv_1_80() {
    #[allow(unused)]
    let _ = 1;
}
