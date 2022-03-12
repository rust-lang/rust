//! Syntax highlighting for escape sequences
use crate::syntax_highlighting::highlights::Highlights;
use crate::{HlRange, HlTag};
use syntax::ast::IsString;
use syntax::TextSize;

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
