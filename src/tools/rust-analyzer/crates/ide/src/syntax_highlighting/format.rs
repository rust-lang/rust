//! Syntax highlighting for format macro strings.
use ide_db::{
    SymbolKind,
    defs::Definition,
    syntax_helpers::format_string::{FormatSpecifier, is_format_string, lex_format_specifiers},
};
use span::Edition;
use syntax::{AstToken, ast};

use crate::{
    HlRange, HlTag,
    syntax_highlighting::{highlight::highlight_def, highlights::Highlights},
};

pub(super) fn highlight_format_string(
    stack: &mut Highlights,
    sema: &hir::Semantics<'_, ide_db::RootDatabase>,
    krate: Option<hir::Crate>,
    string: &ast::String,
    expanded_string: &ast::String,
    edition: Edition,
) {
    if is_format_string(expanded_string) {
        let start = string.syntax().text_range().start();
        // FIXME: Replace this with the HIR info we have now.
        lex_format_specifiers(string, &mut |piece_range, kind| {
            if let Some(highlight) = highlight_format_specifier(kind) {
                stack.add(HlRange {
                    range: piece_range + start,
                    highlight: highlight.into(),
                    binding_hash: None,
                });
            }
        });

        return;
    }

    if let Some(parts) = sema.as_format_args_parts(string) {
        parts.into_iter().for_each(|(range, res)| {
            if let Some(res) = res {
                stack.add(HlRange {
                    range,
                    highlight: highlight_def(sema, krate, Definition::from(res), edition, true),
                    binding_hash: None,
                })
            }
        })
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
        FormatSpecifier::Escape => HlTag::EscapeSequence,
    })
}
