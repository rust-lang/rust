use ide_db::base_db::SourceDatabase;
use syntax::TextSize;
use syntax::{
    algo::non_trivia_sibling, ast, AstNode, Direction, SyntaxKind, SyntaxToken, TextRange, T,
};

use crate::{AssistContext, AssistId, AssistKind, Assists};

// Assist: flip_comma
//
// Flips two comma-separated items.
//
// ```
// fn main() {
//     ((1, 2),$0 (3, 4));
// }
// ```
// ->
// ```
// fn main() {
//     ((3, 4), (1, 2));
// }
// ```
pub(crate) fn flip_comma(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    let comma = ctx.find_token_syntax_at_offset(T![,])?;
    let prev = non_trivia_sibling(comma.clone().into(), Direction::Prev)?;
    let next = non_trivia_sibling(comma.clone().into(), Direction::Next)?;
    let (mut prev_text, mut next_text) = (prev.to_string(), next.to_string());
    let (mut prev_range, mut next_range) = (prev.text_range(), next.text_range());

    // Don't apply a "flip" in case of a last comma
    // that typically comes before punctuation
    if next.kind().is_punct() {
        return None;
    }

    // Don't apply a "flip" inside the macro call
    // since macro input are just mere tokens
    if comma.parent_ancestors().any(|it| it.kind() == SyntaxKind::MACRO_CALL) {
        return None;
    }

    if let Some(parent) = comma.parent().and_then(ast::TokenTree::cast) {
        // An attribute. It often contains a path followed by a token tree (e.g. `align(2)`), so we have
        // to be smarter.
        let prev_start =
            match comma.siblings_with_tokens(Direction::Prev).skip(1).find(|it| it.kind() == T![,])
            {
                Some(it) => position_after_token(it.as_token().unwrap()),
                None => position_after_token(&parent.left_delimiter_token()?),
            };
        let prev_end = prev.text_range().end();
        let next_start = next.text_range().start();
        let next_end =
            match comma.siblings_with_tokens(Direction::Next).skip(1).find(|it| it.kind() == T![,])
            {
                Some(it) => position_before_token(it.as_token().unwrap()),
                None => position_before_token(&parent.right_delimiter_token()?),
            };
        prev_range = TextRange::new(prev_start, prev_end);
        next_range = TextRange::new(next_start, next_end);
        let file_text = ctx.db().file_text(ctx.file_id().file_id());
        prev_text = file_text[prev_range].to_owned();
        next_text = file_text[next_range].to_owned();
    }

    acc.add(
        AssistId("flip_comma", AssistKind::RefactorRewrite),
        "Flip comma",
        comma.text_range(),
        |edit| {
            edit.replace(prev_range, next_text);
            edit.replace(next_range, prev_text);
        },
    )
}

fn position_before_token(token: &SyntaxToken) -> TextSize {
    match non_trivia_sibling(token.clone().into(), Direction::Prev) {
        Some(prev_token) => prev_token.text_range().end(),
        None => token.text_range().start(),
    }
}

fn position_after_token(token: &SyntaxToken) -> TextSize {
    match non_trivia_sibling(token.clone().into(), Direction::Next) {
        Some(prev_token) => prev_token.text_range().start(),
        None => token.text_range().end(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::tests::{check_assist, check_assist_not_applicable, check_assist_target};

    #[test]
    fn flip_comma_works_for_function_parameters() {
        check_assist(
            flip_comma,
            r#"fn foo(x: i32,$0 y: Result<(), ()>) {}"#,
            r#"fn foo(y: Result<(), ()>, x: i32) {}"#,
        )
    }

    #[test]
    fn flip_comma_target() {
        check_assist_target(flip_comma, r#"fn foo(x: i32,$0 y: Result<(), ()>) {}"#, ",")
    }

    #[test]
    fn flip_comma_before_punct() {
        // See https://github.com/rust-lang/rust-analyzer/issues/1619
        // "Flip comma" assist shouldn't be applicable to the last comma in enum or struct
        // declaration body.
        check_assist_not_applicable(flip_comma, "pub enum Test { A,$0 }");
        check_assist_not_applicable(flip_comma, "pub struct Test { foo: usize,$0 }");
    }

    #[test]
    fn flip_comma_works() {
        check_assist(
            flip_comma,
            r#"fn main() {((1, 2),$0 (3, 4));}"#,
            r#"fn main() {((3, 4), (1, 2));}"#,
        )
    }

    #[test]
    fn flip_comma_not_applicable_for_macro_input() {
        // "Flip comma" assist shouldn't be applicable inside the macro call
        // See https://github.com/rust-lang/rust-analyzer/issues/7693
        check_assist_not_applicable(flip_comma, r#"bar!(a,$0 b)"#);
    }

    #[test]
    fn flip_comma_attribute() {
        check_assist(
            flip_comma,
            r#"#[repr(align(2),$0 C)] struct Foo;"#,
            r#"#[repr(C, align(2))] struct Foo;"#,
        );
        check_assist(
            flip_comma,
            r#"#[foo(bar, baz(1 + 1),$0 qux, other)] struct Foo;"#,
            r#"#[foo(bar, qux, baz(1 + 1), other)] struct Foo;"#,
        );
    }
}
