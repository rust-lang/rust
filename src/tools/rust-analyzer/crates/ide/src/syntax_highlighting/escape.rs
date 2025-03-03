//! Syntax highlighting for escape sequences
use crate::syntax_highlighting::highlights::Highlights;
use crate::{HlRange, HlTag};
use syntax::ast::{Byte, Char, IsString};
use syntax::{AstToken, TextRange, TextSize};

pub(super) fn highlight_escape_string<T: IsString>(stack: &mut Highlights, string: &T) {
    let text = string.text();
    let start = string.syntax().text_range().start();
    string.escaped_char_ranges(&mut |piece_range, char| {
        if text[piece_range.start().into()..].starts_with('\\') {
            let highlight = match char {
                Ok(_) => HlTag::EscapeSequence,
                Err(_) => HlTag::InvalidEscapeSequence,
            };
            stack.add(HlRange {
                range: piece_range + start,
                highlight: highlight.into(),
                binding_hash: None,
            });
        }
    });
}

pub(super) fn highlight_escape_char(stack: &mut Highlights, char: &Char) {
    if char.value().is_err() {
        // We do not emit invalid escapes highlighting here. The lexer would likely be in a bad
        // state and this token contains junk, since `'` is not a reliable delimiter (consider
        // lifetimes). Nonetheless, parser errors should already be emitted.
        return;
    }

    let text = char.text();
    let Some(text) = text
        .strip_prefix('\'')
        .and_then(|it| it.strip_suffix('\''))
        .filter(|it| it.starts_with('\\'))
    else {
        return;
    };

    let range = TextRange::at(
        char.syntax().text_range().start() + TextSize::from(1),
        TextSize::from(text.len() as u32),
    );
    stack.add(HlRange { range, highlight: HlTag::EscapeSequence.into(), binding_hash: None })
}

pub(super) fn highlight_escape_byte(stack: &mut Highlights, byte: &Byte) {
    if byte.value().is_err() {
        // See `highlight_escape_char` for why no error highlighting here.
        return;
    }

    let text = byte.text();
    let Some(text) = text
        .strip_prefix("b'")
        .and_then(|it| it.strip_suffix('\''))
        .filter(|it| it.starts_with('\\'))
    else {
        return;
    };

    let range = TextRange::at(
        byte.syntax().text_range().start() + TextSize::from(2),
        TextSize::from(text.len() as u32),
    );
    stack.add(HlRange { range, highlight: HlTag::EscapeSequence.into(), binding_hash: None })
}
