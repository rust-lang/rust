use syntax::{algo::non_trivia_sibling, Direction, SyntaxKind, T};

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

    acc.add(
        AssistId("flip_comma", AssistKind::RefactorRewrite),
        "Flip comma",
        comma.text_range(),
        |edit| {
            edit.replace(prev.text_range(), next.to_string());
            edit.replace(next.text_range(), prev.to_string());
        },
    )
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
}
