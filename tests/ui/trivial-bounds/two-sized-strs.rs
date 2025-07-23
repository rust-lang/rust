//! Regression test for #121363
//@ compile-flags: -Zmir-enable-passes=+GVN --crate-type lib
//@ build-pass

#![feature(trivial_bounds)]
#![expect(trivial_bounds)]

#[derive(Debug)]
struct TwoStrs(str, str)
where
    str: Sized;
