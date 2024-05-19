//@ check-pass

#![deny(rustdoc::redundant_explicit_links)]

/// [Vec][std::vec::Vec#examples] should not warn, because it's not actually redundant!
/// [This is just an `Option`][std::option::Option] has different display content to actual link!
pub fn func() {}
