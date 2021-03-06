use syntax::{ast, AstToken, SyntaxKind::STRING};

use crate::{AssistContext, AssistId, AssistKind, Assists};

// Assist: replace_string_with_char
//
// Replace string with char.
//
// ```
// fn main() {
//     find("{$0");
// }
// ```
// ->
// ```
// fn main() {
//     find('{');
// }
// ```
pub(crate) fn replace_string_with_char(acc: &mut Assists, ctx: &AssistContext) -> Option<()> {
    let token = ctx.find_token_syntax_at_offset(STRING).and_then(ast::String::cast)?;
    let value = token.value()?;
    let target = token.syntax().text_range();

    if value.chars().take(2).count() != 1 {
        return None;
    }
    let quote_offets = token.quote_offsets()?;

    acc.add(
        AssistId("replace_string_with_char", AssistKind::RefactorRewrite),
        "Replace string with char",
        target,
        |edit| {
            let (left, right) = quote_offets.quotes;
            edit.replace(left, String::from('\''));
            edit.replace(right, String::from('\''));
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
    let s = "$0c";
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
    let s = "$0c";
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
    fn replace_string_with_char_assist_with_emoji() {
        check_assist(
            replace_string_with_char,
            r#"
fn f() {
    let s = "$0😀";
}
"#,
            r##"
fn f() {
    let s = '😀';
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
    let s = "$0test";
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
    format!($0"x", 92)
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
    find($0"x");
}
"#,
            r##"
fn f() {
    find('x');
}
"##,
        )
    }

    #[test]
    fn replace_string_with_char_newline() {
        check_assist(
            replace_string_with_char,
            r#"
fn f() {
    find($0"\n");
}
"#,
            r##"
fn f() {
    find('\n');
}
"##,
        )
    }

    #[test]
    fn replace_string_with_char_unicode_escape() {
        check_assist(
            replace_string_with_char,
            r#"
fn f() {
    find($0"\u{7FFF}");
}
"#,
            r##"
fn f() {
    find('\u{7FFF}');
}
"##,
        )
    }

    #[test]
    fn replace_raw_string_with_char() {
        check_assist(
            replace_string_with_char,
            r##"
fn f() {
    $0r#"X"#
}
"##,
            r##"
fn f() {
    'X'
}
"##,
        )
    }
}
