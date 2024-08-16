//! A set of utils methods to reuse on other abstraction levels

use crate::SyntaxKind;

#[inline]
pub fn is_raw_identifier(name: &str, edition: parser::Edition) -> bool {
    let is_keyword = SyntaxKind::from_keyword(name, edition).is_some();
    is_keyword && !matches!(name, "self" | "crate" | "super" | "Self")
}
