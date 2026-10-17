//@ check-pass
//@ compile-flags: --crate-type=lib
#![feature(sized_hierarchy)]

// Test demonstrating that elaboration of sizedness bounds works in the simplest cases.

trait Trait {}

fn f<T: Trait + std::marker::PointeeSized>() {
    require_metasized::<T>();
}

fn require_metasized<T: std::marker::MetaSized>() {}
