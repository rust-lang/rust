// check-pass

#![deny(rustdoc::redundant_explicit_links)]

/// [Vec][std::vec::Vec#examples] should not warn, because it's not actually redundant!
pub fn func() {}
