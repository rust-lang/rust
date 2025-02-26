//@ known-bug: #121363
//@ compile-flags: -Zmir-enable-passes=+GVN --crate-type lib

#![feature(trivial_bounds)]

#[derive(Debug)]
struct TwoStrs(str, str)
where
    str: Sized;
