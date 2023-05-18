//! Implementation of incremental re-parsing.
//!
//! We use two simple strategies for this:
//!   - if the edit modifies only a single token (like changing an identifier's
//!     letter), we replace only this token.
//!   - otherwise, we search for the nearest `{}` block which contains the edit
//!     and try to parse only this block.

use parser::Reparser;
use text_edit::Indel;

use crate::{
    parsing::build_tree,
    syntax_node::{GreenNode, GreenToken, NodeOrToken, SyntaxElement, SyntaxNode},
    SyntaxError,
    SyntaxKind::*,
    TextRange, TextSize, T,
};

pub(crate) fn incremental_reparse(
    node: &SyntaxNode,
    edit: &Indel,
    errors: Vec<SyntaxError>,
) -> Option<(GreenNode, Vec<SyntaxError>, TextRange)> {
    if let Some((green, new_errors, old_range)) = reparse_token(node, edit) {
        return Some((green, merge_errors(errors, new_errors, old_range, edit), old_range));
    }

    if let Some((green, new_errors, old_range)) = reparse_block(node, edit) {
        return Some((green, merge_errors(errors, new_errors, old_range, edit), old_range));
    }
    None
}

fn reparse_token(
    root: &SyntaxNode,
    edit: &Indel,
) -> Option<(GreenNode, Vec<SyntaxError>, TextRange)> {
    let prev_token = root.covering_element(edit.delete).as_token()?.clone();
    let prev_token_kind = prev_token.kind();
    match prev_token_kind {
        WHITESPACE | COMMENT | IDENT | STRING | BYTE_STRING | C_STRING => {
            if prev_token_kind == WHITESPACE || prev_token_kind == COMMENT {
                // removing a new line may extends previous token
                let deleted_range = edit.delete - prev_token.text_range().start();
                if prev_token.text()[deleted_range].contains('\n') {
                    return None;
                }
            }

            let mut new_text = get_text_after_edit(prev_token.clone().into(), edit);
            let (new_token_kind, new_err) = parser::LexedStr::single_token(&new_text)?;

            if new_token_kind != prev_token_kind
                || (new_token_kind == IDENT && is_contextual_kw(&new_text))
            {
                return None;
            }

            // Check that edited token is not a part of the bigger token.
            // E.g. if for source code `bruh"str"` the user removed `ruh`, then
            // `b` no longer remains an identifier, but becomes a part of byte string literal
            if let Some(next_char) = root.text().char_at(prev_token.text_range().end()) {
                new_text.push(next_char);
                let token_with_next_char = parser::LexedStr::single_token(&new_text);
                if let Some((_kind, _error)) = token_with_next_char {
                    return None;
                }
                new_text.pop();
            }

            let new_token = GreenToken::new(rowan::SyntaxKind(prev_token_kind.into()), &new_text);
            let range = TextRange::up_to(TextSize::of(&new_text));
            Some((
                prev_token.replace_with(new_token),
                new_err.into_iter().map(|msg| SyntaxError::new(msg, range)).collect(),
                prev_token.text_range(),
            ))
        }
        _ => None,
    }
}

fn reparse_block(
    root: &SyntaxNode,
    edit: &Indel,
) -> Option<(GreenNode, Vec<SyntaxError>, TextRange)> {
    let (node, reparser) = find_reparsable_node(root, edit.delete)?;
    let text = get_text_after_edit(node.clone().into(), edit);

    let lexed = parser::LexedStr::new(text.as_str());
    let parser_input = lexed.to_input();
    if !is_balanced(&lexed) {
        return None;
    }

    let tree_traversal = reparser.parse(&parser_input);

    let (green, new_parser_errors, _eof) = build_tree(lexed, tree_traversal);

    Some((node.replace_with(green), new_parser_errors, node.text_range()))
}

fn get_text_after_edit(element: SyntaxElement, edit: &Indel) -> String {
    let edit = Indel::replace(edit.delete - element.text_range().start(), edit.insert.clone());

    let mut text = match element {
        NodeOrToken::Token(token) => token.text().to_string(),
        NodeOrToken::Node(node) => node.text().to_string(),
    };
    edit.apply(&mut text);
    text
}

fn is_contextual_kw(text: &str) -> bool {
    matches!(text, "auto" | "default" | "union")
}

fn find_reparsable_node(node: &SyntaxNode, range: TextRange) -> Option<(SyntaxNode, Reparser)> {
    let node = node.covering_element(range);

    node.ancestors().find_map(|node| {
        let first_child = node.first_child_or_token().map(|it| it.kind());
        let parent = node.parent().map(|it| it.kind());
        Reparser::for_node(node.kind(), first_child, parent).map(|r| (node, r))
    })
}

fn is_balanced(lexed: &parser::LexedStr<'_>) -> bool {
    if lexed.is_empty() || lexed.kind(0) != T!['{'] || lexed.kind(lexed.len() - 1) != T!['}'] {
        return false;
    }
    let mut balance = 0usize;
    for i in 1..lexed.len() - 1 {
        match lexed.kind(i) {
            T!['{'] => balance += 1,
            T!['}'] => {
                balance = match balance.checked_sub(1) {
                    Some(b) => b,
                    None => return false,
                }
            }
            _ => (),
        }
    }
    balance == 0
}

fn merge_errors(
    old_errors: Vec<SyntaxError>,
    new_errors: Vec<SyntaxError>,
    range_before_reparse: TextRange,
    edit: &Indel,
) -> Vec<SyntaxError> {
    let mut res = Vec::new();

    for old_err in old_errors {
        let old_err_range = old_err.range();
        if old_err_range.end() <= range_before_reparse.start() {
            res.push(old_err);
        } else if old_err_range.start() >= range_before_reparse.end() {
            let inserted_len = TextSize::of(&edit.insert);
            res.push(old_err.with_range((old_err_range + inserted_len) - edit.delete.len()));
            // Note: extra parens are intentional to prevent uint underflow, HWAB (here was a bug)
        }
    }
    res.extend(new_errors.into_iter().map(|new_err| {
        // fighting borrow checker with a variable ;)
        let offsetted_range = new_err.range() + range_before_reparse.start();
        new_err.with_range(offsetted_range)
    }));
    res
}

#[cfg(test)]
mod tests {
    use test_utils::{assert_eq_text, extract_range};

    use super::*;
    use crate::{AstNode, Parse, SourceFile};

    fn do_check(before: &str, replace_with: &str, reparsed_len: u32) {
        let (range, before) = extract_range(before);
        let edit = Indel::replace(range, replace_with.to_owned());
        let after = {
            let mut after = before.clone();
            edit.apply(&mut after);
            after
        };

        let fully_reparsed = SourceFile::parse(&after);
        let incrementally_reparsed: Parse<SourceFile> = {
            let before = SourceFile::parse(&before);
            let (green, new_errors, range) =
                incremental_reparse(before.tree().syntax(), &edit, before.errors.to_vec()).unwrap();
            assert_eq!(range.len(), reparsed_len.into(), "reparsed fragment has wrong length");
            Parse::new(green, new_errors)
        };

        assert_eq_text!(
            &format!("{:#?}", fully_reparsed.tree().syntax()),
            &format!("{:#?}", incrementally_reparsed.tree().syntax()),
        );
        assert_eq!(fully_reparsed.errors(), incrementally_reparsed.errors());
    }

    #[test] // FIXME: some test here actually test token reparsing
    fn reparse_block_tests() {
        do_check(
            r"
fn foo() {
    let x = foo + $0bar$0
}
",
            "baz",
            3,
        );
        do_check(
            r"
fn foo() {
    let x = foo$0 + bar$0
}
",
            "baz",
            25,
        );
        do_check(
            r"
struct Foo {
    f: foo$0$0
}
",
            ",\n    g: (),",
            14,
        );
        do_check(
            r"
fn foo {
    let;
    1 + 1;
    $092$0;
}
",
            "62",
            31, // FIXME: reparse only int literal here
        );
        do_check(
            r"
mod foo {
    fn $0$0
}
",
            "bar",
            11,
        );

        do_check(
            r"
trait Foo {
    type $0Foo$0;
}
",
            "Output",
            3,
        );
        do_check(
            r"
impl IntoIterator<Item=i32> for Foo {
    f$0$0
}
",
            "n next(",
            9,
        );
        do_check(r"use a::b::{foo,$0,bar$0};", "baz", 10);
        do_check(
            r"
pub enum A {
    Foo$0$0
}
",
            "\nBar;\n",
            11,
        );
        do_check(
            r"
foo!{a, b$0$0 d}
",
            ", c[3]",
            8,
        );
        do_check(
            r"
fn foo() {
    vec![$0$0]
}
",
            "123",
            14,
        );
        do_check(
            r"
extern {
    fn$0;$0
}
",
            " exit(code: c_int)",
            11,
        );
    }

    #[test]
    fn reparse_token_tests() {
        do_check(
            r"$0$0
fn foo() -> i32 { 1 }
",
            "\n\n\n   \n",
            1,
        );
        do_check(
            r"
fn foo() -> $0$0 {}
",
            "  \n",
            2,
        );
        do_check(
            r"
fn $0foo$0() -> i32 { 1 }
",
            "bar",
            3,
        );
        do_check(
            r"
fn foo$0$0foo() {  }
",
            "bar",
            6,
        );
        do_check(
            r"
fn foo /* $0$0 */ () {}
",
            "some comment",
            6,
        );
        do_check(
            r"
fn baz $0$0 () {}
",
            "    \t\t\n\n",
            2,
        );
        do_check(
            r"
fn baz $0$0 () {}
",
            "    \t\t\n\n",
            2,
        );
        do_check(
            r"
/// foo $0$0omment
mod { }
",
            "c",
            14,
        );
        do_check(
            r#"
fn -> &str { "Hello$0$0" }
"#,
            ", world",
            7,
        );
        do_check(
            r#"
fn -> &str { // "Hello$0$0"
"#,
            ", world",
            10,
        );
        do_check(
            r##"
fn -> &str { r#"Hello$0$0"#
"##,
            ", world",
            10,
        );
        do_check(
            r"
#[derive($0Copy$0)]
enum Foo {

}
",
            "Clone",
            4,
        );
    }

    #[test]
    fn reparse_str_token_with_error_unchanged() {
        do_check(r#""$0Unclosed$0 string literal"#, "Still unclosed", 24);
    }

    #[test]
    fn reparse_str_token_with_error_fixed() {
        do_check(r#""unterminated$0$0"#, "\"", 13);
    }

    #[test]
    fn reparse_block_with_error_in_middle_unchanged() {
        do_check(
            r#"fn main() {
                if {}
                32 + 4$0$0
                return
                if {}
            }"#,
            "23",
            105,
        )
    }

    #[test]
    fn reparse_block_with_error_in_middle_fixed() {
        do_check(
            r#"fn main() {
                if {}
                32 + 4$0$0
                return
                if {}
            }"#,
            ";",
            105,
        )
    }
}
