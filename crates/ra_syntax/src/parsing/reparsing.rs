//! Implementation of incremental re-parsing.
//!
//! We use two simple strategies for this:
//!   - if the edit modifies only a single token (like changing an identifier's
//!     letter), we replace only this token.
//!   - otherwise, we search for the nearest `{}` block which contains the edit
//!     and try to parse only this block.

use ra_text_edit::AtomTextEdit;
use ra_parser::Reparser;

use crate::{
    SyntaxKind::*, TextRange, TextUnit, SyntaxError,
    algo,
    syntax_node::{GreenNode, SyntaxNode, GreenToken, SyntaxElement},
    parsing::{
        text_token_source::TextTokenSource,
        text_tree_sink::TextTreeSink,
        lexer::{tokenize, Token},
    },
    T,
};

pub(crate) fn incremental_reparse(
    node: &SyntaxNode,
    edit: &AtomTextEdit,
    errors: Vec<SyntaxError>,
) -> Option<(GreenNode, Vec<SyntaxError>, TextRange)> {
    if let Some((green, old_range)) = reparse_token(node, &edit) {
        return Some((green, merge_errors(errors, Vec::new(), old_range, edit), old_range));
    }

    if let Some((green, new_errors, old_range)) = reparse_block(node, &edit) {
        return Some((green, merge_errors(errors, new_errors, old_range, edit), old_range));
    }
    None
}

fn reparse_token<'node>(
    root: &'node SyntaxNode,
    edit: &AtomTextEdit,
) -> Option<(GreenNode, TextRange)> {
    let token = algo::find_covering_element(root, edit.delete).as_token()?;
    match token.kind() {
        WHITESPACE | COMMENT | IDENT | STRING | RAW_STRING => {
            if token.kind() == WHITESPACE || token.kind() == COMMENT {
                // removing a new line may extends previous token
                if token.text().to_string()[edit.delete - token.range().start()].contains('\n') {
                    return None;
                }
            }

            let text = get_text_after_edit(token.into(), &edit);
            let lex_tokens = tokenize(&text);
            let lex_token = match lex_tokens[..] {
                [lex_token] if lex_token.kind == token.kind() => lex_token,
                _ => return None,
            };

            if lex_token.kind == IDENT && is_contextual_kw(&text) {
                return None;
            }

            if let Some(next_char) = root.text().char_at(token.range().end()) {
                let tokens_with_next_char = tokenize(&format!("{}{}", text, next_char));
                if tokens_with_next_char.len() == 1 {
                    return None;
                }
            }

            let new_token = GreenToken::new(rowan::SyntaxKind(token.kind().into()), text.into());
            Some((token.replace_with(new_token), token.range()))
        }
        _ => None,
    }
}

fn reparse_block<'node>(
    root: &'node SyntaxNode,
    edit: &AtomTextEdit,
) -> Option<(GreenNode, Vec<SyntaxError>, TextRange)> {
    let (node, reparser) = find_reparsable_node(root, edit.delete)?;
    let text = get_text_after_edit(node.into(), &edit);
    let tokens = tokenize(&text);
    if !is_balanced(&tokens) {
        return None;
    }
    let mut token_source = TextTokenSource::new(&text, &tokens);
    let mut tree_sink = TextTreeSink::new(&text, &tokens);
    reparser.parse(&mut token_source, &mut tree_sink);
    let (green, new_errors) = tree_sink.finish();
    Some((node.replace_with(green), new_errors, node.range()))
}

fn get_text_after_edit(element: SyntaxElement, edit: &AtomTextEdit) -> String {
    let edit = AtomTextEdit::replace(edit.delete - element.range().start(), edit.insert.clone());
    let text = match element {
        SyntaxElement::Token(token) => token.text().to_string(),
        SyntaxElement::Node(node) => node.text().to_string(),
    };
    edit.apply(text)
}

fn is_contextual_kw(text: &str) -> bool {
    match text {
        "auto" | "default" | "union" => true,
        _ => false,
    }
}

fn find_reparsable_node(node: &SyntaxNode, range: TextRange) -> Option<(&SyntaxNode, Reparser)> {
    let node = algo::find_covering_element(node, range);
    let mut ancestors = match node {
        SyntaxElement::Token(it) => it.parent().ancestors(),
        SyntaxElement::Node(it) => it.ancestors(),
    };
    ancestors.find_map(|node| {
        let first_child = node.first_child_or_token().map(|it| it.kind());
        let parent = node.parent().map(|it| it.kind());
        Reparser::for_node(node.kind(), first_child, parent).map(|r| (node, r))
    })
}

fn is_balanced(tokens: &[Token]) -> bool {
    if tokens.is_empty()
        || tokens.first().unwrap().kind != T!['{']
        || tokens.last().unwrap().kind != T!['}']
    {
        return false;
    }
    let mut balance = 0usize;
    for t in &tokens[1..tokens.len() - 1] {
        match t.kind {
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
    old_range: TextRange,
    edit: &AtomTextEdit,
) -> Vec<SyntaxError> {
    let mut res = Vec::new();
    for e in old_errors {
        if e.offset() <= old_range.start() {
            res.push(e)
        } else if e.offset() >= old_range.end() {
            res.push(e.add_offset(TextUnit::of_str(&edit.insert), edit.delete.len()));
        }
    }
    for e in new_errors {
        res.push(e.add_offset(old_range.start(), 0.into()));
    }
    res
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use test_utils::{extract_range, assert_eq_text};

    use crate::{SourceFile, AstNode, Parse};
    use super::*;

    fn do_check(before: &str, replace_with: &str, reparsed_len: u32) {
        let (range, before) = extract_range(before);
        let edit = AtomTextEdit::replace(range, replace_with.to_owned());
        let after = edit.apply(before.clone());

        let fully_reparsed = SourceFile::parse2(&after);
        let incrementally_reparsed = {
            let f = SourceFile::parse(&before);
            let edit = AtomTextEdit { delete: range, insert: replace_with.to_string() };
            let (green, new_errors, range) =
                incremental_reparse(f.syntax(), &edit, f.errors()).unwrap();
            assert_eq!(range.len(), reparsed_len.into(), "reparsed fragment has wrong length");
            Parse { tree: SourceFile::new(green), errors: Arc::new(new_errors) }
        };

        assert_eq_text!(
            &fully_reparsed.tree.syntax().debug_dump(),
            &incrementally_reparsed.tree.syntax().debug_dump(),
        );
    }

    #[test] // FIXME: some test here actually test token reparsing
    fn reparse_block_tests() {
        do_check(
            r"
fn foo() {
    let x = foo + <|>bar<|>
}
",
            "baz",
            3,
        );
        do_check(
            r"
fn foo() {
    let x = foo<|> + bar<|>
}
",
            "baz",
            25,
        );
        do_check(
            r"
struct Foo {
    f: foo<|><|>
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
    <|>92<|>;
}
",
            "62",
            31, // FIXME: reparse only int literal here
        );
        do_check(
            r"
mod foo {
    fn <|><|>
}
",
            "bar",
            11,
        );

        do_check(
            r"
trait Foo {
    type <|>Foo<|>;
}
",
            "Output",
            3,
        );
        do_check(
            r"
impl IntoIterator<Item=i32> for Foo {
    f<|><|>
}
",
            "n next(",
            9,
        );
        do_check(r"use a::b::{foo,<|>,bar<|>};", "baz", 10);
        do_check(
            r"
pub enum A {
    Foo<|><|>
}
",
            "\nBar;\n",
            11,
        );
        do_check(
            r"
foo!{a, b<|><|> d}
",
            ", c[3]",
            8,
        );
        do_check(
            r"
fn foo() {
    vec![<|><|>]
}
",
            "123",
            14,
        );
        do_check(
            r"
extern {
    fn<|>;<|>
}
",
            " exit(code: c_int)",
            11,
        );
    }

    #[test]
    fn reparse_token_tests() {
        do_check(
            r"<|><|>
fn foo() -> i32 { 1 }
",
            "\n\n\n   \n",
            1,
        );
        do_check(
            r"
fn foo() -> <|><|> {}
",
            "  \n",
            2,
        );
        do_check(
            r"
fn <|>foo<|>() -> i32 { 1 }
",
            "bar",
            3,
        );
        do_check(
            r"
fn foo<|><|>foo() {  }
",
            "bar",
            6,
        );
        do_check(
            r"
fn foo /* <|><|> */ () {}
",
            "some comment",
            6,
        );
        do_check(
            r"
fn baz <|><|> () {}
",
            "    \t\t\n\n",
            2,
        );
        do_check(
            r"
fn baz <|><|> () {}
",
            "    \t\t\n\n",
            2,
        );
        do_check(
            r"
/// foo <|><|>omment
mod { }
",
            "c",
            14,
        );
        do_check(
            r#"
fn -> &str { "Hello<|><|>" }
"#,
            ", world",
            7,
        );
        do_check(
            r#"
fn -> &str { // "Hello<|><|>"
"#,
            ", world",
            10,
        );
        do_check(
            r##"
fn -> &str { r#"Hello<|><|>"#
"##,
            ", world",
            10,
        );
        do_check(
            r"
#[derive(<|>Copy<|>)]
enum Foo {

}
",
            "Clone",
            4,
        );
    }
}
