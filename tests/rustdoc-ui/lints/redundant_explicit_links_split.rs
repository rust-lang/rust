#![deny(rustdoc::redundant_explicit_links)]

// Right now, redundant_explicit_links won't produce a warning at all if
// rustdoc isn't able to calculate an accurate span for the link.
//
// If that changes and this test starts to fail, you should add the
// appropriate annotations, and, also, make sure that it doesn't
// suggest a correction that wipes out the `fn` formal signature
// or the `#[inline]` attribute.

/// [std::clone::Clone](
pub fn split_outer_inner() {
    //! std::clone::Clone)
}

/// [std::clone::Clone](std::clone::Clone
pub fn split_outer_inner_b() {
    //! )
}

/// [std::clone::Clone](
#[inline]
/// std::clone::Clone)
pub fn split_attr() {
}

/// [std::clone::Clone](std::clone::Clone
#[inline]
/// )
pub fn split_attr_b() {
}

/// [std::clone::Clone](
/// std::clone::Clone)
//~^^ ERROR redundant_explicit_links
pub fn not_split() {
}

/// [std::clone::Clone](std::clone::Clone
/// )
//~^^ ERROR redundant_explicit_links
pub fn not_split_b() {
}
