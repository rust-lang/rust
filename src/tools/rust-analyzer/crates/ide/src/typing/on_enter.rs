//! Handles the `Enter` key press. At the momently, this only continues
//! comments, but should handle indent some time in the future as well.

use ide_db::base_db::RootQueryDb;
use ide_db::{FilePosition, RootDatabase};
use syntax::{
    AstNode, SmolStr, SourceFile,
    SyntaxKind::*,
    SyntaxNode, SyntaxToken, TextRange, TextSize, TokenAtOffset,
    algo::find_node_at_offset,
    ast::{self, AstToken, edit::IndentLevel},
};

use ide_db::text_edit::TextEdit;

// Feature: On Enter
//
// rust-analyzer can override <kbd>Enter</kbd> key to make it smarter:
//
// - <kbd>Enter</kbd> inside triple-slash comments automatically inserts `///`
// - <kbd>Enter</kbd> in the middle or after a trailing space in `//` inserts `//`
// - <kbd>Enter</kbd> inside `//!` doc comments automatically inserts `//!`
// - <kbd>Enter</kbd> after `{` indents contents and closing `}` of single-line block
//
// This action needs to be assigned to shortcut explicitly.
//
// Note that, depending on the other installed extensions, this feature can visibly slow down typing.
// Similarly, if rust-analyzer crashes or stops responding, `Enter` might not work.
// In that case, you can still press `Shift-Enter` to insert a newline.
//
// #### VS Code
//
// Add the following to `keybindings.json`:
// ```json
// {
//   "key": "Enter",
//   "command": "rust-analyzer.onEnter",
//   "when": "editorTextFocus && !suggestWidgetVisible && editorLangId == rust"
// }
// ````
//
// When using the Vim plugin:
// ```json
// {
//   "key": "Enter",
//   "command": "rust-analyzer.onEnter",
//   "when": "editorTextFocus && !suggestWidgetVisible && editorLangId == rust && vim.mode == 'Insert'"
// }
// ````
//
// ![On Enter](https://user-images.githubusercontent.com/48062697/113065578-04c21800-91b1-11eb-82b8-22b8c481e645.gif)
pub(crate) fn on_enter(db: &RootDatabase, position: FilePosition) -> Option<TextEdit> {
    let editioned_file_id_wrapper =
        ide_db::base_db::EditionedFileId::current_edition(db, position.file_id);
    let parse = db.parse(editioned_file_id_wrapper);
    let file = parse.tree();
    let token = file.syntax().token_at_offset(position.offset).left_biased()?;

    if let Some(comment) = ast::Comment::cast(token.clone()) {
        return on_enter_in_comment(&comment, &file, position.offset);
    }

    if token.kind() == L_CURLY {
        // Typing enter after the `{` of a block expression, where the `}` is on the same line
        if let Some(edit) = find_node_at_offset(file.syntax(), position.offset - TextSize::of('{'))
            .and_then(|block| on_enter_in_block(block, position))
        {
            cov_mark::hit!(indent_block_contents);
            return Some(edit);
        }

        // Typing enter after the `{` of a use tree list.
        if let Some(edit) = find_node_at_offset(file.syntax(), position.offset - TextSize::of('{'))
            .and_then(|list| on_enter_in_use_tree_list(list, position))
        {
            cov_mark::hit!(indent_block_contents);
            return Some(edit);
        }
    }

    None
}

fn on_enter_in_comment(
    comment: &ast::Comment,
    file: &ast::SourceFile,
    offset: TextSize,
) -> Option<TextEdit> {
    if comment.kind().shape.is_block() {
        return None;
    }

    let prefix = comment.prefix();
    let comment_range = comment.syntax().text_range();
    if offset < comment_range.start() + TextSize::of(prefix) {
        return None;
    }

    let mut remove_trailing_whitespace = false;
    // Continuing single-line non-doc comments (like this one :) ) is annoying
    if prefix == "//" && comment_range.end() == offset {
        if comment.text().ends_with(' ') {
            cov_mark::hit!(continues_end_of_line_comment_with_space);
            remove_trailing_whitespace = true;
        } else if !followed_by_comment(comment) {
            return None;
        }
    }

    let indent = node_indent(file, comment.syntax())?;
    let inserted = format!("\n{indent}{prefix} $0");
    let delete = if remove_trailing_whitespace {
        let trimmed_len = comment.text().trim_end().len() as u32;
        let trailing_whitespace_len = comment.text().len() as u32 - trimmed_len;
        TextRange::new(offset - TextSize::from(trailing_whitespace_len), offset)
    } else {
        TextRange::empty(offset)
    };
    let edit = TextEdit::replace(delete, inserted);
    Some(edit)
}

fn on_enter_in_block(block: ast::BlockExpr, position: FilePosition) -> Option<TextEdit> {
    let contents = block_contents(&block)?;

    if block.syntax().text().contains_char('\n') {
        return None;
    }

    let indent = IndentLevel::from_node(block.syntax());
    let mut edit = TextEdit::insert(position.offset, format!("\n{}$0", indent + 1));
    edit.union(TextEdit::insert(contents.text_range().end(), format!("\n{indent}"))).ok()?;
    Some(edit)
}

fn on_enter_in_use_tree_list(list: ast::UseTreeList, position: FilePosition) -> Option<TextEdit> {
    if list.syntax().text().contains_char('\n') {
        return None;
    }

    let indent = IndentLevel::from_node(list.syntax());
    let mut edit = TextEdit::insert(position.offset, format!("\n{}$0", indent + 1));
    edit.union(TextEdit::insert(list.r_curly_token()?.text_range().start(), format!("\n{indent}")))
        .ok()?;
    Some(edit)
}

fn block_contents(block: &ast::BlockExpr) -> Option<SyntaxNode> {
    let mut node = block.tail_expr().map(|e| e.syntax().clone());

    for stmt in block.statements() {
        if node.is_some() {
            // More than 1 node in the block
            return None;
        }

        node = Some(stmt.syntax().clone());
    }

    node
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
        let (analysis, position) = fixture::position(before);
        let result = analysis.on_enter(position).unwrap()?;

        let mut actual = analysis.file_text(position.file_id).unwrap().to_string();
        result.apply(&mut actual);
        Some(actual)
    }

    fn do_check(
        #[rust_analyzer::rust_fixture] ra_fixture_before: &str,
        #[rust_analyzer::rust_fixture] ra_fixture_after: &str,
    ) {
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
    fn continues_another_doc_comment() {
        do_check(
            r#"
fn main() {
    //! Documentation for$0 on enter
    let x = 1 + 1;
}
"#,
            r#"
fn main() {
    //! Documentation for
    //! $0 on enter
    let x = 1 + 1;
}
"#,
        );
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

    #[test]
    fn indents_fn_body_block() {
        cov_mark::check!(indent_block_contents);
        do_check(
            r#"
fn f() {$0()}
        "#,
            r#"
fn f() {
    $0()
}
        "#,
        );
    }

    #[test]
    fn indents_block_expr() {
        do_check(
            r#"
fn f() {
    let x = {$0()};
}
        "#,
            r#"
fn f() {
    let x = {
        $0()
    };
}
        "#,
        );
    }

    #[test]
    fn indents_match_arm() {
        do_check(
            r#"
fn f() {
    match 6 {
        1 => {$0f()},
        _ => (),
    }
}
        "#,
            r#"
fn f() {
    match 6 {
        1 => {
            $0f()
        },
        _ => (),
    }
}
        "#,
        );
    }

    #[test]
    fn indents_block_with_statement() {
        do_check(
            r#"
fn f() {$0a = b}
        "#,
            r#"
fn f() {
    $0a = b
}
        "#,
        );
        do_check(
            r#"
fn f() {$0fn f() {}}
        "#,
            r#"
fn f() {
    $0fn f() {}
}
        "#,
        );
    }

    #[test]
    fn indents_nested_blocks() {
        do_check(
            r#"
fn f() {$0{}}
        "#,
            r#"
fn f() {
    $0{}
}
        "#,
        );
    }

    #[test]
    fn does_not_indent_empty_block() {
        do_check_noop(
            r#"
fn f() {$0}
        "#,
        );
        do_check_noop(
            r#"
fn f() {{$0}}
        "#,
        );
    }

    #[test]
    fn does_not_indent_block_with_too_much_content() {
        do_check_noop(
            r#"
fn f() {$0 a = b; ()}
        "#,
        );
        do_check_noop(
            r#"
fn f() {$0 a = b; a = b; }
        "#,
        );
    }

    #[test]
    fn does_not_indent_multiline_block() {
        do_check_noop(
            r#"
fn f() {$0
}
        "#,
        );
        do_check_noop(
            r#"
fn f() {$0

}
        "#,
        );
    }

    #[test]
    fn indents_use_tree_list() {
        do_check(
            r#"
use crate::{$0};
            "#,
            r#"
use crate::{
    $0
};
            "#,
        );
        do_check(
            r#"
use crate::{$0Object, path::to::OtherThing};
            "#,
            r#"
use crate::{
    $0Object, path::to::OtherThing
};
            "#,
        );
        do_check(
            r#"
use {crate::{$0Object, path::to::OtherThing}};
            "#,
            r#"
use {crate::{
    $0Object, path::to::OtherThing
}};
            "#,
        );
        do_check(
            r#"
use {
    crate::{$0Object, path::to::OtherThing}
};
            "#,
            r#"
use {
    crate::{
        $0Object, path::to::OtherThing
    }
};
            "#,
        );
    }

    #[test]
    fn does_not_indent_use_tree_list_when_not_at_curly_brace() {
        do_check_noop(
            r#"
use path::{Thing$0};
            "#,
        );
    }

    #[test]
    fn does_not_indent_use_tree_list_without_curly_braces() {
        do_check_noop(
            r#"
use path::Thing$0;
            "#,
        );
        do_check_noop(
            r#"
use path::$0Thing;
            "#,
        );
        do_check_noop(
            r#"
use path::Thing$0};
            "#,
        );
        do_check_noop(
            r#"
use path::{$0Thing;
            "#,
        );
    }

    #[test]
    fn does_not_indent_multiline_use_tree_list() {
        do_check_noop(
            r#"
use path::{$0
    Thing
};
            "#,
        );
    }
}
