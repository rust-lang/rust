#![deny(rustdoc::redundant_explicit_links)]
use std::clone::Clone;

/// [Clone](
pub fn split_outer_inner() {
    //! std::clone::Clone)
//~^^^ ERROR redundant_explicit_links
}

/// [Clone](std::clone::Clone
pub fn split_outer_inner_b() {
    //! )
//~^^^ ERROR redundant_explicit_links
}

/// [Clone](
#[inline]
/// std::clone::Clone)
//~^^^ ERROR redundant_explicit_links
pub fn split_attr() {
}

/// [Clone](std::clone::Clone
#[inline]
/// )
//~^^^ ERROR redundant_explicit_links
pub fn split_attr_b() {
}

/** [Clone]( */ #[inline] /** std::clone::Clone) */
//~^ ERROR redundant_explicit_links
pub fn split_blockcomment_attr() {
}

/** [Clone](std::clone::Clone */ #[inline] /** ) */
//~^ ERROR redundant_explicit_links
pub fn split_blockcomment_attr_b() {
}

/** [Clone]( */
#[inline]
/** std::clone::Clone) */
//~^^^ ERROR redundant_explicit_links
pub fn split_blockcomment_multiline_attr() {
}

/** [Clone](std::clone::Clone */
#[inline]
/** ) */
//~^^^ ERROR redundant_explicit_links
pub fn split_blockcomment_multiline_attr_b() {
}

/** [Clone]( */ #[inline]
/** std::clone::Clone) */
//~^^ ERROR redundant_explicit_links
pub fn split_blockcomment_twoline_attr() {
}

/** [Clone](std::clone::Clone */ #[inline]
/** ) */
//~^^ ERROR redundant_explicit_links
pub fn split_blockcomment_twoline_attr_b() {
}

/** [Clone]( */
#[inline] /** std::clone::Clone) */
//~^^ ERROR redundant_explicit_links
pub fn split_blockcomment_secondline_attr() {
}

/** [Clone](std::clone::Clone */
#[inline] /** ) */
//~^^ ERROR redundant_explicit_links
pub fn split_blockcomment_secondline_attr_b() {
}

/** [Clone](std::clone::Clone) */ #[inline]
//~^ ERROR redundant_explicit_links
//~| SUGGESTION [Clone]
pub fn split_blockcomment_singleline_attr() {
}

/** [Clone](std::clone::Clone) */ #[inline] pub fn split_blockcomment_singleline_attr_b() { }
//~^ ERROR redundant_explicit_links
//~| SUGGESTION [Clone]

/// [Clone](
/// std::clone::Clone)
//~^^ ERROR redundant_explicit_links
//~| SUGGESTION [Clone]
pub fn not_split() {
}

/// [Clone](std::clone::Clone
/// )
//~^^ ERROR redundant_explicit_links
//~| SUGGESTION [Clone]
pub fn not_split_b() {
}
