//! Syntax highlighting for format macro strings.
use syntax::{
    ast::{self, FormatSpecifier, HasFormatSpecifier},
    AstNode, AstToken, SyntaxElement, SyntaxKind, SyntaxNode, TextRange,
};

use crate::{HlRange, HlTag, SymbolKind};

use super::highlights::Highlights;

#[derive(Default)]
pub(super) struct FormatStringHighlighter {
    format_string: Option<SyntaxElement>,
}

impl FormatStringHighlighter {
    pub(super) fn check_for_format_string(&mut self, parent: &SyntaxNode) {
        // Check if macro takes a format string and remember it for highlighting later.
        // The macros that accept a format string expand to a compiler builtin macros
        // `format_args` and `format_args_nl`.
        if let Some(name) = parent
            .parent()
            .and_then(ast::MacroCall::cast)
            .and_then(|mc| mc.path())
            .and_then(|p| p.segment())
            .and_then(|s| s.name_ref())
        {
            match name.text().as_str() {
                "format_args" | "format_args_nl" => {
                    self.format_string = parent
                        .children_with_tokens()
                        .filter(|t| t.kind() != SyntaxKind::WHITESPACE)
                        .nth(1)
                        .filter(|e| ast::String::can_cast(e.kind()))
                }
                _ => {}
            }
        }
    }
    pub(super) fn highlight_format_string(
        &self,
        stack: &mut Highlights,
        string: &impl HasFormatSpecifier,
        range: TextRange,
    ) {
        if self.format_string.as_ref() == Some(&SyntaxElement::from(string.syntax().clone())) {
            string.lex_format_specifier(|piece_range, kind| {
                if let Some(highlight) = highlight_format_specifier(kind) {
                    stack.add(HlRange {
                        range: piece_range + range.start(),
                        highlight: highlight.into(),
                        binding_hash: None,
                    });
                }
            });
        }
    }
}

fn highlight_format_specifier(kind: FormatSpecifier) -> Option<HlTag> {
    Some(match kind {
        FormatSpecifier::Open
        | FormatSpecifier::Close
        | FormatSpecifier::Colon
        | FormatSpecifier::Fill
        | FormatSpecifier::Align
        | FormatSpecifier::Sign
        | FormatSpecifier::NumberSign
        | FormatSpecifier::DollarSign
        | FormatSpecifier::Dot
        | FormatSpecifier::Asterisk
        | FormatSpecifier::QuestionMark => HlTag::FormatSpecifier,
        FormatSpecifier::Integer | FormatSpecifier::Zero => HlTag::NumericLiteral,
        FormatSpecifier::Identifier => HlTag::Symbol(SymbolKind::Local),
    })
}
