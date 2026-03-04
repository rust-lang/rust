//@ aux-build:rmeta-meta.rs
//@ no-prefer-dynamic
//@ build-fail

// Check that we do not ICE when we need optimized MIR but it is missing.

extern crate rmeta_meta;

fn main() {
    rmeta_meta::missing_optimized_mir();
}

//~? ERROR crate `rmeta_meta` required to be available in rlib format, but was not found in this form
//~? ERROR missing optimized MIR for `missing_optimized_mir` in the crate `rmeta_meta`
