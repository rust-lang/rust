//@ known-bug: #121363
//@ compile-flags: -Zmir-opt-level=5 --crate-type lib

#![feature(trivial_bounds)]

#[derive(Debug)]
struct TwoStrs(str, str)
where
    str: Sized;
