//@ check-pass
//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

#![feature(checked_type_aliases)]

use std::any::Any;

type Coerce = Box<dyn Any>;

fn test() -> Coerce {
    Box::new(1)
}

fn main() {}
