//! FIXME: write short doc here

use ra_syntax::{SourceFile, TextUnit};
use ra_text_edit::TextEdit;

pub use test_utils::*;

pub fn check_action<F: Fn(&SourceFile, TextUnit) -> Option<TextEdit>>(
    before: &str,
    after: &str,
    f: F,
) {
    let (before_cursor_pos, before) = extract_offset(before);
    let file = SourceFile::parse(&before).ok().unwrap();
    let result = f(&file, before_cursor_pos).expect("code action is not applicable");
    let actual = result.apply(&before);
    let actual_cursor_pos =
        result.apply_to_offset(before_cursor_pos).expect("cursor position is affected by the edit");
    let actual = add_cursor(&actual, actual_cursor_pos);
    assert_eq_text!(after, &actual);
}
