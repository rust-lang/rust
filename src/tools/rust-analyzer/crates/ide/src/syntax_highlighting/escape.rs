//! Syntax highlighting for escape sequences
use crate::syntax_highlighting::highlights::Highlights;
use crate::{HlRange, HlTag};
use syntax::ast::{Byte, Char, IsString};
use syntax::{AstToken, TextRange, TextSize};

pub(super) fn highlight_escape_string<T: IsString>(
    stack: &mut Highlights,
    string: &T,
    start: TextSize,
) {
    string.escaped_char_ranges(&mut |piece_range, char| {
        if string.text()[piece_range.start().into()..].starts_with('\\') {
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

pub(super) fn highlight_escape_char(stack: &mut Highlights, char: &Char, start: TextSize) {
    if char.value().is_none() {
        // We do not emit invalid escapes highlighting here. The lexer would likely be in a bad
        // state and this token contains junks, since `'` is not a reliable delimiter (consider
        // lifetimes). Nonetheless, parser errors should already be emitted.
        return;
    }

    let text = char.text();
    if !text.starts_with('\'') || !text.ends_with('\'') {
        return;
    }

    let text = &text[1..text.len() - 1];
    if !text.starts_with('\\') {
        return;
    }

    let range =
        TextRange::new(start + TextSize::from(1), start + TextSize::from(text.len() as u32 + 1));
    stack.add(HlRange { range, highlight: HlTag::EscapeSequence.into(), binding_hash: None })
}

pub(super) fn highlight_escape_byte(stack: &mut Highlights, byte: &Byte, start: TextSize) {
    if byte.value().is_none() {
        // See `highlight_escape_char` for why no error highlighting here.
        return;
    }

    let text = byte.text();
    if !text.starts_with("b'") || !text.ends_with('\'') {
        return;
    }

    let text = &text[2..text.len() - 1];
    if !text.starts_with('\\') {
        return;
    }

    let range =
        TextRange::new(start + TextSize::from(2), start + TextSize::from(text.len() as u32 + 2));
    stack.add(HlRange { range, highlight: HlTag::EscapeSequence.into(), binding_hash: None })
}
