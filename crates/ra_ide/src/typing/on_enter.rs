//! Handles the `Enter` key press. At the momently, this only continues
//! comments, but should handle indent some time in the future as well.

use ra_db::{FilePosition, SourceDatabase};
use ra_ide_db::RootDatabase;
use ra_syntax::{
    ast::{self, AstToken},
    AstNode, SmolStr, SourceFile,
    SyntaxKind::*,
    SyntaxToken, TextUnit, TokenAtOffset,
};
use ra_text_edit::TextEdit;

use crate::{SourceChange, SourceFileEdit};

pub(crate) fn on_enter(db: &RootDatabase, position: FilePosition) -> Option<SourceChange> {
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
    if position.offset < comment_range.start() + TextUnit::of_str(prefix) {
        return None;
    }

    // Continuing single-line non-doc comments (like this one :) ) is annoying
    if prefix == "//" && comment_range.end() == position.offset && !followed_by_comment(&comment) {
        return None;
    }

    let indent = node_indent(&file, comment.syntax())?;
    let inserted = format!("\n{}{} ", indent, prefix);
    let cursor_position = position.offset + TextUnit::of_str(&inserted);
    let edit = TextEdit::insert(position.offset, inserted);

    Some(
        SourceChange::source_file_edit(
            "on enter",
            SourceFileEdit { edit, file_id: position.file_id },
        )
        .with_cursor(FilePosition { offset: cursor_position, file_id: position.file_id }),
    )
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
    use test_utils::{add_cursor, assert_eq_text, extract_offset};

    use crate::mock_analysis::single_file;

    use super::*;

    fn apply_on_enter(before: &str) -> Option<String> {
        let (offset, before) = extract_offset(before);
        let (analysis, file_id) = single_file(&before);
        let result = analysis.on_enter(FilePosition { offset, file_id }).unwrap()?;

        assert_eq!(result.source_file_edits.len(), 1);
        let actual = result.source_file_edits[0].edit.apply(&before);
        let actual = add_cursor(&actual, result.cursor_position.unwrap().offset);
        Some(actual)
    }

    fn do_check(ra_fixture_before: &str, ra_fixture_after: &str) {
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
/// Some docs<|>
fn foo() {
}
",
            r"
/// Some docs
/// <|>
fn foo() {
}
",
        );

        do_check(
            r"
impl S {
    /// Some<|> docs.
    fn foo() {}
}
",
            r"
impl S {
    /// Some
    /// <|> docs.
    fn foo() {}
}
",
        );

        do_check(
            r"
///<|> Some docs
fn foo() {
}
",
            r"
///
/// <|> Some docs
fn foo() {
}
",
        );
    }

    #[test]
    fn does_not_continue_before_doc_comment() {
        do_check_noop(r"<|>//! docz");
    }

    #[test]
    fn continues_code_comment_in_the_middle_of_line() {
        do_check(
            r"
fn main() {
    // Fix<|> me
    let x = 1 + 1;
}
",
            r"
fn main() {
    // Fix
    // <|> me
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
    // Fix<|>
    // me
    let x = 1 + 1;
}
",
            r"
fn main() {
    // Fix
    // <|>
    // me
    let x = 1 + 1;
}
",
        );
    }

    #[test]
    fn does_not_continue_end_of_code_comment() {
        do_check_noop(
            r"
fn main() {
    // Fix me<|>
    let x = 1 + 1;
}
",
        );
    }
}
