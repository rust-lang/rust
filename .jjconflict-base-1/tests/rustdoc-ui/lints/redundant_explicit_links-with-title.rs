//@ check-pass

#![deny(rustdoc::redundant_explicit_links)]

/// [drop](drop "This function is not magic")
///
/// This link should not lint, because it specifies a link title, and it is
/// not possible to remove the explicit link without also removing the title.
///
/// [Vec][vec]
///
/// [vec]: std::vec::Vec "A contiguous growable array type"
///
/// This also applies to reference-style links.
pub fn func() {}
