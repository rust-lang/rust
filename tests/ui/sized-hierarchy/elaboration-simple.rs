//@ check-pass
//@ compile-flags: --crate-type=lib
#![feature(sized_hierarchy)]

// Test demonstrating that elaboration of sizedness bounds works in the simplest cases.

trait Trait {}

fn f<T: Trait + std::marker::PointeeSized>() {
    require_sizeofval::<T>();
}

fn require_sizeofval<T: std::marker::SizeOfVal>() {}
