//@ check-pass

#![deny(rustdoc::redundant_explicit_links)]

/// [Vec][std::vec::Vec#examples] should not warn, because it's not actually redundant!
/// [This is just an `Option`][std::option::Option] has different display content to actual link!
pub fn func() {}

// Regression guard for https://github.com/rust-lang/rust/issues/155458.
/// [NoRedundancyTarget](struct.NoRedundancyTarget.html#fragment) should not warn.
pub struct NoRedundancySource;

pub struct NoRedundancyTarget;
