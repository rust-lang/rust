//@ only-wasm32
//@ compile-flags: -C panic=abort
//@ build-pass

// Test that a `-C panic=abort` binary crate can link to a `-C panic=unwind` core.

fn main() {}
