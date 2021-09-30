//! Syntax highlighting for format macro strings.
use ide_db::SymbolKind;
use syntax::{
    ast::{self, FormatSpecifier, HasFormatSpecifier},
    AstNode, AstToken, TextRange,
};

use crate::{syntax_highlighting::highlights::Highlights, HlRange, HlTag};

pub(super) fn highlight_format_string(
    stack: &mut Highlights,
    string: &ast::String,
    range: TextRange,
) {
    if is_format_string(string).is_none() {
        return;
    }

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

fn is_format_string(string: &ast::String) -> Option<()> {
    let parent = string.syntax().parent()?;

    let name = parent.parent().and_then(ast::MacroCall::cast)?.path()?.segment()?.name_ref()?;
    if !matches!(
        name.text().as_str(),
        "format_args" | "format_args_nl" | "const_format_args" | "panic_2015" | "panic_2021"
    ) {
        return None;
    }

    // NB: we match against `panic_2015`/`panic_2021` here because they have a special-cased arm for
    // `"{}"`, which otherwise wouldn't get highlighted.

    let first_literal = parent
        .children_with_tokens()
        .find_map(|it| it.as_token().cloned().and_then(ast::String::cast))?;
    if &first_literal != string {
        return None;
    }

    Some(())
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
