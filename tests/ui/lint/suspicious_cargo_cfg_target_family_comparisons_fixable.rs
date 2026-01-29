// Test fixable suggestions for the `suspicious_cargo_cfg_target_family_comparisons` lint.

//@ check-pass
//@ exec-env:CARGO_CFG_TARGET_FAMILY=unix
//@ run-rustfix

use std::env;

fn main() {
    let target_family = env::var("CARGO_CFG_TARGET_FAMILY").unwrap();
    let unix = "unix";

    if target_family == unix {}
    //~^ WARN comparing against `CARGO_CFG_TARGET_FAMILY` directly may break in the future

    if target_family != unix {}
    //~^ WARN comparing against `CARGO_CFG_TARGET_FAMILY` directly may break in the future

    let _ = env::var("CARGO_CFG_TARGET_FAMILY").unwrap() == "unix";
    //~^ WARN comparing against `CARGO_CFG_TARGET_FAMILY` directly may break in the future
}
