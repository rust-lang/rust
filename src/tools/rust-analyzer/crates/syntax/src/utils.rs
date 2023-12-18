//! A set of utils methods to reuse on other abstraction levels

use crate::SyntaxKind;

pub fn is_raw_identifier(name: &str) -> bool {
    let is_keyword = SyntaxKind::from_keyword(name).is_some();
    is_keyword && !matches!(name, "self" | "crate" | "super" | "Self")
}
