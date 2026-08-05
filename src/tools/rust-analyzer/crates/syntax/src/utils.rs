//! A set of utils methods to reuse on other abstraction levels

use crate::SyntaxKind;
use rustc_lexer;

#[inline]
/// Checks that the name is an identifier.
/// This also means that it is not a strict keyword.
/// But it may be a weak keyword.
pub fn is_identifier(name: &str, edition: parser::Edition) -> bool {
    if rustc_lexer::is_ident(name) {
        if let Some(syntax_kind) = SyntaxKind::from_keyword(name, edition)
            && syntax_kind.is_strict_keyword(edition)
        {
            false
        } else {
            true
        }
    } else {
        false
    }
}

#[inline]
pub fn is_raw_identifier(name: &str, edition: parser::Edition) -> bool {
    let is_keyword = SyntaxKind::from_keyword(name, edition).is_some();
    is_keyword && !matches!(name, "self" | "crate" | "super" | "Self")
}
