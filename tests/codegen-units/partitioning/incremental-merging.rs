//@ incremental
//@ compile-flags: -Copt-level=0 -Ccodegen-units=3

#![crate_type = "rlib"]

// This test makes sure that merging of CGUs works together with incremental
// compilation but at the same time does not modify names of CGUs that were not
// affected by merging.
//
// CGU partitioning creates one CGU per module, so with 4 modules and codegen-units=3,
// two of the modules should be merged. We expect CGUs `aaa` and `bbb` to be merged
// (because they are the smallest), while `ccc` and `ddd` should stay untouched.

pub mod aaa {
    //~ MONO_ITEM fn aaa::foo @@ incremental_merging-aaa--incremental_merging-bbb[External]
    pub fn foo(a: u64) -> u64 {
        a + 1
    }
}

pub mod bbb {
    //~ MONO_ITEM fn bbb::foo @@ incremental_merging-aaa--incremental_merging-bbb[External]
    pub fn foo(a: u64, b: u64) -> u64 {
        a + b + 1
    }
}

pub mod ccc {
    //~ MONO_ITEM fn ccc::foo @@ incremental_merging-ccc[External]
    pub fn foo(a: u64, b: u64, c: u64) -> u64 {
        a + b + c + 1
    }
}

pub mod ddd {
    //~ MONO_ITEM fn ddd::foo @@ incremental_merging-ddd[External]
    pub fn foo(a: u64, b: u64, c: u64, d: u64) -> u64 {
        a + b + c + d + 1
    }
}
