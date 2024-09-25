//@ error-pattern: circular modules
// Regression test for #97589: a doc-comment on a circular module bypassed cycle detection

#![crate_type = "lib"]

pub mod recursive;
