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
    expanded_string: &ast::String,
    range: TextRange,
) {
    if is_format_string(expanded_string).is_none() {
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
    // Check if `string` is a format string argument of a macro invocation.
    // `string` is a string literal, mapped down into the innermost macro expansion.
    // Since `format_args!` etc. remove the format string when expanding, but place all arguments
    // in the expanded output, we know that the string token is (part of) the format string if it
    // appears in `format_args!` (otherwise it would have been mapped down further).
    //
    // This setup lets us correctly highlight the components of `concat!("{}", "bla")` format
    // strings. It still fails for `concat!("{", "}")`, but that is rare.

    let macro_call = string.syntax().ancestors().find_map(ast::MacroCall::cast)?;
    let name = macro_call.path()?.segment()?.name_ref()?;

    if !matches!(
        name.text().as_str(),
        "format_args" | "format_args_nl" | "const_format_args" | "panic_2015" | "panic_2021"
    ) {
        return None;
    }

    // NB: we match against `panic_2015`/`panic_2021` here because they have a special-cased arm for
    // `"{}"`, which otherwise wouldn't get highlighted.

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
