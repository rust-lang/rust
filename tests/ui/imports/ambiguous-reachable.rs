//@ build-fail
//@ aux-crate: ambiguous_reachable_extern=ambiguous-reachable-extern.rs

#![allow(ambiguous_glob_imports)]

fn main() {
    ambiguous_reachable_extern::generic::<u8>();
}

//~? ERROR missing optimized MIR
