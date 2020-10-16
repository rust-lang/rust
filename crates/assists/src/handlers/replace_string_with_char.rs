use std::borrow::Cow;

use syntax::{
    ast::{self, HasQuotes, HasStringValue},
    AstToken,
    SyntaxKind::{RAW_STRING, STRING},
    TextRange, TextSize,
};
use test_utils::mark;

use crate::{AssistContext, AssistId, AssistKind, Assists};

// Assist: replace_string_with_char
//
// Replace string with char
//
// ```
// fn main() {
//     find("{<|>");
// }
// ```
// ->
// ```
// fn main() {
//     find('{');
// }
// ```
pub(crate) fn replace_string_with_char(acc: &mut Assists, ctx: &AssistContext) -> Option<()> {
    let token = ctx.find_token_at_offset(STRING).and_then(ast::String::cast)?;
    let value = token.value()?;
    let target = token.syntax().text_range();
    if value.len() > 1 || value.is_empty() {
        return None;
    }

    acc.add(
        AssistId("replace_string_with_char", AssistKind::RefactorRewrite),
        "Replace string with char",
        target,
        |edit| {
            edit.replace(token.syntax().text_range(), format!("'{}'", value));
        },
    )
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_assist, check_assist_not_applicable, check_assist_target};

    use super::*;

    #[test]
    fn replace_string_with_char_target() {
        check_assist_target(
            replace_string_with_char,
            r#"
            fn f() {
                let s = "<|>c";
            }
            "#,
            r#""c""#,
        );
    }

    #[test]
    fn replace_string_with_char_assist() {
        check_assist(
            replace_string_with_char,
            r#"
    fn f() {
        let s = "<|>c";
    }
    "#,
            r##"
    fn f() {
        let s = 'c';
    }
    "##,
        )
    }

    #[test]
    fn replace_string_with_char_assist_not_applicable() {
        check_assist_not_applicable(
            replace_string_with_char,
            r#"
    fn f() {
        let s = "<|>test";
    }
    "#,
        )
    }

    #[test]
    fn replace_string_with_char_works_inside_macros() {
        check_assist(
            replace_string_with_char,
            r#"
                fn f() {
                    format!(<|>"x", 92)
                }
                "#,
            r##"
                fn f() {
                    format!('x', 92)
                }
                "##,
        )
    }

    #[test]
    fn replace_string_with_char_works_func_args() {
        check_assist(
            replace_string_with_char,
            r#"
                fn f() {
                    find(<|>"x");
                }
                "#,
            r##"
                fn f() {
                    find('x');
                }
                "##,
        )
    }
}
