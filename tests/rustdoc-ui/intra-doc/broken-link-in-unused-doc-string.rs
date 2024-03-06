// Test that we don't ICE with broken links that don't show up in the docs.

//@ check-pass
//@ edition: 2021

/// [1]
//~^ WARN unresolved link to `1`
//~| WARN unresolved link to `1`
pub use {std, core};

/// [2]
pub use {};

/// [3]
//~^ WARN unresolved link to `3`
pub extern crate alloc;
