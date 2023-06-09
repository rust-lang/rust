//! Syntax highlighting for escape sequences
use crate::syntax_highlighting::highlights::Highlights;
use crate::{HlRange, HlTag};
use syntax::ast::{Char, IsString};
use syntax::{AstToken, TextRange, TextSize};

pub(super) fn highlight_escape_string<T: IsString>(
    stack: &mut Highlights,
    string: &T,
    start: TextSize,
) {
    string.escaped_char_ranges(&mut |piece_range, char| {
        if char.is_err() {
            return;
        }

        if string.text()[piece_range.start().into()..].starts_with('\\') {
            stack.add(HlRange {
                range: piece_range + start,
                highlight: HlTag::EscapeSequence.into(),
                binding_hash: None,
            });
        }
    });
}

pub(super) fn highlight_escape_char(stack: &mut Highlights, char: &Char, start: TextSize) {
    if char.value().is_none() {
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
