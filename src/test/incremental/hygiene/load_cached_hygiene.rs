// revisions:rpass1 rpass2
// compile-flags: -Z query-dep-graph
// aux-build:cached_hygiene.rs

// This tests the following scenario
// 1. A foreign crate is compiled with incremental compilation.
//    This causes hygiene information to be saved to the incr cache.
// 2. One function is the foreign crate is modified. This causes the
//    optimized mir for an unmodified function to be loaded from the
//    incremental cache and written out to the crate metadata.
// 3. In the process of loading and writing out this function's MIR,
//    we load hygiene information from the incremental cache and
//    write it to our metadata.
// 4. This hygiene information is loaded by another crate (this file)

// Previously, this situation would cause hygiene identifiers
// (SyntaxContexts and ExpnIds) to get corrupted when we tried to
// serialize the hygiene information loaded from the incr cache into
// the metadata. Specifically, we were not resetting `orig_id`
// for an `EpxnData` generate in the current crate, which would cause
// us to serialize the `ExpnId` pointing to a garbage location in
// the metadata.

#![feature(rustc_attrs)]

#![rustc_partition_reused(module="load_cached_hygiene-call_unchanged_function", cfg="rpass2")]
#![rustc_partition_codegened(module="load_cached_hygiene-call_changed_function", cfg="rpass2")]


extern crate cached_hygiene;

pub mod call_unchanged_function {

    pub fn unchanged() {
        cached_hygiene::unchanged_fn();
    }
}

pub mod call_changed_function {
    pub fn changed() {
        cached_hygiene::changed_fn();
    }
}

pub fn main() {
    call_unchanged_function::unchanged();
    call_changed_function::changed();
}
