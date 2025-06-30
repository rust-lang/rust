//@ check-pass
//@ compile-flags: --crate-type=lib
#![feature(sized_hierarchy)]

trait Trait {}

fn f<T: Trait + std::marker::PointeeSized>() {}
