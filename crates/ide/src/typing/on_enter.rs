//! Handles the `Enter` key press. At the momently, this only continues
//! comments, but should handle indent some time in the future as well.

use ide_db::base_db::{FilePosition, SourceDatabase};
use ide_db::RootDatabase;
use syntax::{
    ast::{self, AstToken},
    AstNode, SmolStr, SourceFile,
    SyntaxKind::*,
    SyntaxToken, TextRange, TextSize, TokenAtOffset,
};

use text_edit::TextEdit;

// Feature: On Enter
//
// rust-analyzer can override kbd:[Enter] key to make it smarter:
//
// - kbd:[Enter] inside triple-slash comments automatically inserts `///`
// - kbd:[Enter] in the middle or after a trailing space in `//` inserts `//`
//
// This action needs to be assigned to shortcut explicitly.
//
// VS Code::
//
// Add the following to `keybindings.json`:
// [source,json]
// ----
// {
//   "key": "Enter",
//   "command": "rust-analyzer.onEnter",
//   "when": "editorTextFocus && !suggestWidgetVisible && editorLangId == rust"
// }
// ----
//
// image::https://user-images.githubusercontent.com/48062697/113065578-04c21800-91b1-11eb-82b8-22b8c481e645.gif[]
pub(crate) fn on_enter(db: &RootDatabase, position: FilePosition) -> Option<TextEdit> {
    let parse = db.parse(position.file_id);
    let file = parse.tree();
    let comment = file
        .syntax()
        .token_at_offset(position.offset)
        .left_biased()
        .and_then(ast::Comment::cast)?;

    if comment.kind().shape.is_block() {
        return None;
    }

    let prefix = comment.prefix();
    let comment_range = comment.syntax().text_range();
    if position.offset < comment_range.start() + TextSize::of(prefix) {
        return None;
    }

    let mut remove_trailing_whitespace = false;
    // Continuing single-line non-doc comments (like this one :) ) is annoying
    if prefix == "//" && comment_range.end() == position.offset {
        if comment.text().ends_with(' ') {
            cov_mark::hit!(continues_end_of_line_comment_with_space);
            remove_trailing_whitespace = true;
        } else if !followed_by_comment(&comment) {
            return None;
        }
    }

    let indent = node_indent(&file, comment.syntax())?;
    let inserted = format!("\n{}{} $0", indent, prefix);
    let delete = if remove_trailing_whitespace {
        let trimmed_len = comment.text().trim_end().len() as u32;
        let trailing_whitespace_len = comment.text().len() as u32 - trimmed_len;
        TextRange::new(position.offset - TextSize::from(trailing_whitespace_len), position.offset)
    } else {
        TextRange::empty(position.offset)
    };
    let edit = TextEdit::replace(delete, inserted);
    Some(edit)
}

fn followed_by_comment(comment: &ast::Comment) -> bool {
    let ws = match comment.syntax().next_token().and_then(ast::Whitespace::cast) {
        Some(it) => it,
        None => return false,
    };
    if ws.spans_multiple_lines() {
        return false;
    }
    ws.syntax().next_token().and_then(ast::Comment::cast).is_some()
}

fn node_indent(file: &SourceFile, token: &SyntaxToken) -> Option<SmolStr> {
    let ws = match file.syntax().token_at_offset(token.text_range().start()) {
        TokenAtOffset::Between(l, r) => {
            assert!(r == *token);
            l
        }
        TokenAtOffset::Single(n) => {
            assert!(n == *token);
            return Some("".into());
        }
        TokenAtOffset::None => unreachable!(),
    };
    if ws.kind() != WHITESPACE {
        return None;
    }
    let text = ws.text();
    let pos = text.rfind('\n').map(|it| it + 1).unwrap_or(0);
    Some(text[pos..].into())
}

#[cfg(test)]
mod tests {
    use stdx::trim_indent;
    use test_utils::assert_eq_text;

    use crate::fixture;

    fn apply_on_enter(before: &str) -> Option<String> {
        let (analysis, position) = fixture::position(&before);
        let result = analysis.on_enter(position).unwrap()?;

        let mut actual = analysis.file_text(position.file_id).unwrap().to_string();
        result.apply(&mut actual);
        Some(actual)
    }

    fn do_check(ra_fixture_before: &str, ra_fixture_after: &str) {
        let ra_fixture_after = &trim_indent(ra_fixture_after);
        let actual = apply_on_enter(ra_fixture_before).unwrap();
        assert_eq_text!(ra_fixture_after, &actual);
    }

    fn do_check_noop(ra_fixture_text: &str) {
        assert!(apply_on_enter(ra_fixture_text).is_none())
    }

    #[test]
    fn continues_doc_comment() {
        do_check(
            r"
/// Some docs$0
fn foo() {
}
",
            r"
/// Some docs
/// $0
fn foo() {
}
",
        );

        do_check(
            r"
impl S {
    /// Some$0 docs.
    fn foo() {}
}
",
            r"
impl S {
    /// Some
    /// $0 docs.
    fn foo() {}
}
",
        );

        do_check(
            r"
///$0 Some docs
fn foo() {
}
",
            r"
///
/// $0 Some docs
fn foo() {
}
",
        );
    }

    #[test]
    fn does_not_continue_before_doc_comment() {
        do_check_noop(r"$0//! docz");
    }

    #[test]
    fn continues_code_comment_in_the_middle_of_line() {
        do_check(
            r"
fn main() {
    // Fix$0 me
    let x = 1 + 1;
}
",
            r"
fn main() {
    // Fix
    // $0 me
    let x = 1 + 1;
}
",
        );
    }

    #[test]
    fn continues_code_comment_in_the_middle_several_lines() {
        do_check(
            r"
fn main() {
    // Fix$0
    // me
    let x = 1 + 1;
}
",
            r"
fn main() {
    // Fix
    // $0
    // me
    let x = 1 + 1;
}
",
        );
    }

    #[test]
    fn does_not_continue_end_of_line_comment() {
        do_check_noop(
            r"
fn main() {
    // Fix me$0
    let x = 1 + 1;
}
",
        );
    }

    #[test]
    fn continues_end_of_line_comment_with_space() {
        cov_mark::check!(continues_end_of_line_comment_with_space);
        do_check(
            r#"
fn main() {
    // Fix me $0
    let x = 1 + 1;
}
"#,
            r#"
fn main() {
    // Fix me
    // $0
    let x = 1 + 1;
}
"#,
        );
    }

    #[test]
    fn trims_all_trailing_whitespace() {
        do_check(
            "
fn main() {
    // Fix me  \t\t   $0
    let x = 1 + 1;
}
",
            "
fn main() {
    // Fix me
    // $0
    let x = 1 + 1;
}
",
        );
    }
}
