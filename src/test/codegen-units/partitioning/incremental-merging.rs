// ignore-tidy-linelength
// We specify -C incremental here because we want to test the partitioning for
// incremental compilation
// compile-flags:-Zprint-mono-items=lazy -Cincremental=tmp/partitioning-tests/incremental-merging
// compile-flags:-Ccodegen-units=3

#![crate_type = "rlib"]

// This test makes sure that merging of CGUs works together with incremental
// compilation but at the same time does not modify names of CGUs that were not
// affected by merging.
//
// We expect CGUs `aaa` and `bbb` to be merged (because they are the smallest),
// while `ccc` and `ddd` are supposed to stay untouched.

pub mod aaa {
    //~ MONO_ITEM fn incremental_merging::aaa[0]::foo[0] @@ incremental_merging-aaa--incremental_merging-bbb[External]
    pub fn foo(a: u64) -> u64 {
        a + 1
    }
}

pub mod bbb {
    //~ MONO_ITEM fn incremental_merging::bbb[0]::foo[0] @@ incremental_merging-aaa--incremental_merging-bbb[External]
    pub fn foo(a: u64, b: u64) -> u64 {
        a + b + 1
    }
}

pub mod ccc {
    //~ MONO_ITEM fn incremental_merging::ccc[0]::foo[0] @@ incremental_merging-ccc[External]
    pub fn foo(a: u64, b: u64, c: u64) -> u64 {
        a + b + c + 1
    }
}

pub mod ddd {
    //~ MONO_ITEM fn incremental_merging::ddd[0]::foo[0] @@ incremental_merging-ddd[External]
    pub fn foo(a: u64, b: u64, c: u64, d: u64) -> u64 {
        a + b + c + d + 1
    }
}
