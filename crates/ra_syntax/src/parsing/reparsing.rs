//! Implementation of incremental re-parsing.
//!
//! We use two simple strategies for this:
//!   - if the edit modifies only a single token (like changing an identifier's
//!     letter), we replace only this token.
//!   - otherwise, we search for the nearest `{}` block which contains the edit
//!     and try to parse only this block.

use ra_parser::Reparser;
use ra_text_edit::AtomTextEdit;

use crate::{
    algo,
    parsing::{
        lexer::{single_token, tokenize, ParsedTokens, Token},
        text_token_source::TextTokenSource,
        text_tree_sink::TextTreeSink,
    },
    syntax_node::{GreenNode, GreenToken, NodeOrToken, SyntaxElement, SyntaxNode},
    SyntaxError,
    SyntaxKind::*,
    TextRange, TextUnit, T,
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
    let prev_token = algo::find_covering_element(root, edit.delete).as_token()?.clone();
    let prev_token_kind = prev_token.kind();
    match prev_token_kind {
        WHITESPACE | COMMENT | IDENT | STRING | RAW_STRING => {
            if prev_token_kind == WHITESPACE || prev_token_kind == COMMENT {
                // removing a new line may extends previous token
                let deleted_range = edit.delete - prev_token.text_range().start();
                if prev_token.text()[deleted_range].contains('\n') {
                    return None;
                }
            }

            let mut new_text = get_text_after_edit(prev_token.clone().into(), &edit);
            let new_token_kind = single_token(&new_text)?.token.kind;

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
                let token_with_next_char = single_token(&new_text);
                if token_with_next_char.is_some() {
                    return None;
                }
                new_text.pop();
            }

            let new_token =
                GreenToken::new(rowan::SyntaxKind(prev_token_kind.into()), new_text.into());
            Some((prev_token.replace_with(new_token), prev_token.text_range()))
        }
        _ => None,
    }
}

fn reparse_block<'node>(
    root: &'node SyntaxNode,
    edit: &AtomTextEdit,
) -> Option<(GreenNode, Vec<SyntaxError>, TextRange)> {
    let (node, reparser) = find_reparsable_node(root, edit.delete)?;
    let text = get_text_after_edit(node.clone().into(), &edit);
    let ParsedTokens { tokens, errors } = tokenize(&text);
    if !is_balanced(&tokens) {
        return None;
    }
    let mut token_source = TextTokenSource::new(&text, &tokens);
    let mut tree_sink = TextTreeSink::new(&text, &tokens, errors);
    reparser.parse(&mut token_source, &mut tree_sink);
    let (green, new_errors) = tree_sink.finish();
    Some((node.replace_with(green), new_errors, node.text_range()))
}

fn get_text_after_edit(element: SyntaxElement, edit: &AtomTextEdit) -> String {
    let edit =
        AtomTextEdit::replace(edit.delete - element.text_range().start(), edit.insert.clone());

    // Note: we could move this match to a method or even further: use enum_dispatch crate
    // https://crates.io/crates/enum_dispatch
    let text = match element {
        NodeOrToken::Token(token) => token.text().to_string(),
        NodeOrToken::Node(node) => node.text().to_string(),
    };
    edit.apply(text)
}

fn is_contextual_kw(text: &str) -> bool {
    match text {
        "auto" | "default" | "union" => true,
        _ => false,
    }
}

fn find_reparsable_node(node: &SyntaxNode, range: TextRange) -> Option<(SyntaxNode, Reparser)> {
    let node = algo::find_covering_element(node, range);

    // Note: we could move this match to a method or even further: use enum_dispatch crate
    // https://crates.io/crates/enum_dispatch
    let mut ancestors = match node {
        NodeOrToken::Token(it) => it.parent().ancestors(),
        NodeOrToken::Node(it) => it.ancestors(),
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
    use test_utils::{assert_eq_text, extract_range};

    use super::*;
    use crate::{AstNode, Parse, SourceFile};

    fn do_check(before: &str, replace_with: &str, reparsed_len: u32) {
        let (range, before) = extract_range(before);
        let edit = AtomTextEdit::replace(range, replace_with.to_owned());
        let after = edit.apply(before.clone());

        let fully_reparsed = SourceFile::parse(&after);
        let incrementally_reparsed: Parse<SourceFile> = {
            let f = SourceFile::parse(&before);
            // FIXME: it seems this initialization statement is unnecessary (see edit in outer scope)
            // Investigate whether it should really be removed.
            let edit = AtomTextEdit { delete: range, insert: replace_with.to_string() };
            let (green, new_errors, range) =
                incremental_reparse(f.tree().syntax(), &edit, f.errors.to_vec()).unwrap();
            assert_eq!(range.len(), reparsed_len.into(), "reparsed fragment has wrong length");
            Parse::new(green, new_errors)
        };

        assert_eq_text!(
            &format!("{:#?}", fully_reparsed.tree().syntax()),
            &format!("{:#?}", incrementally_reparsed.tree().syntax()),
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
