use syntax::{
    AstToken,
    SyntaxKind::{CHAR, STRING},
    TextRange, TextSize, ast,
    ast::IsString,
};

use crate::{AssistContext, AssistId, Assists, utils::string_suffix};

// Assist: replace_string_with_char
//
// Replace string literal with char literal.
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
pub(crate) fn replace_string_with_char(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    let token = ctx.find_token_syntax_at_offset(STRING).and_then(ast::String::cast)?;
    let value = token.value().ok()?;
    let target = token.syntax().text_range();

    if value.chars().take(2).count() != 1 {
        return None;
    }
    let quote_offsets = token.quote_offsets()?;

    acc.add(
        AssistId::refactor_rewrite("replace_string_with_char"),
        "Replace string with char",
        target,
        |edit| {
            let (left, right) = quote_offsets.quotes;
            let suffix = TextSize::of(string_suffix(token.text()).unwrap_or_default());
            let right = TextRange::new(right.start(), right.end() - suffix);
            edit.replace(left, '\'');
            edit.replace(right, '\'');
            if token.text_without_quotes() == "'" {
                edit.insert(left.end(), '\\');
            }
        },
    )
}

// Assist: replace_char_with_string
//
// Replace a char literal with a string literal.
//
// ```
// fn main() {
//     find('{$0');
// }
// ```
// ->
// ```
// fn main() {
//     find("{");
// }
// ```
pub(crate) fn replace_char_with_string(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    let token = ctx.find_token_syntax_at_offset(CHAR)?;
    let target = token.text_range();

    acc.add(
        AssistId::refactor_rewrite("replace_char_with_string"),
        "Replace char with string",
        target,
        |edit| {
            let suffix = string_suffix(token.text()).unwrap_or_default();
            if token.text().starts_with("'\"'") {
                edit.replace(token.text_range(), format!(r#""\""{suffix}"#));
            } else {
                let len = TextSize::of('\'');
                let suffix = TextSize::of(suffix);
                edit.replace(TextRange::at(target.start(), len), '"');
                edit.replace(TextRange::at(target.end() - suffix - len, len), '"');
            }
        },
    )
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_assist, check_assist_not_applicable};

    use super::*;

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
    fn replace_string_with_char_has_suffix() {
        check_assist(
            replace_string_with_char,
            r#"
fn f() {
    let s = "$0c"i32;
}
"#,
            r##"
fn f() {
    let s = 'c'i32;
}
"##,
        )
    }

    #[test]
    fn replace_string_with_char_assist_with_multi_byte_char() {
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
    fn replace_string_with_char_multiple_chars() {
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

    #[test]
    fn replace_char_with_string_assist() {
        check_assist(
            replace_char_with_string,
            r"
fn f() {
    let s = '$0c';
}
",
            r#"
fn f() {
    let s = "c";
}
"#,
        )
    }

    #[test]
    fn replace_char_with_string_assist_with_multi_byte_char() {
        check_assist(
            replace_char_with_string,
            r"
fn f() {
    let s = '$0😀';
}
",
            r#"
fn f() {
    let s = "😀";
}
"#,
        )
    }

    #[test]
    fn replace_char_with_string_newline() {
        check_assist(
            replace_char_with_string,
            r"
fn f() {
    find($0'\n');
}
",
            r#"
fn f() {
    find("\n");
}
"#,
        )
    }

    #[test]
    fn replace_char_with_string_unicode_escape() {
        check_assist(
            replace_char_with_string,
            r"
fn f() {
    find($0'\u{7FFF}');
}
",
            r#"
fn f() {
    find("\u{7FFF}");
}
"#,
        )
    }

    #[test]
    fn replace_char_with_string_quote() {
        check_assist(
            replace_char_with_string,
            r#"
fn f() {
    find($0'"');
}
"#,
            r#"
fn f() {
    find("\"");
}
"#,
        )
    }

    #[test]
    fn replace_char_with_string_quote_has_suffix() {
        check_assist(
            replace_char_with_string,
            r#"
fn f() {
    find($0'"'i32);
}
"#,
            r#"
fn f() {
    find("\""i32);
}
"#,
        )
    }

    #[test]
    fn replace_char_with_string_escaped_quote_has_suffix() {
        check_assist(
            replace_char_with_string,
            r#"
fn f() {
    find($0'\"'i32);
}
"#,
            r#"
fn f() {
    find("\""i32);
}
"#,
        )
    }

    #[test]
    fn replace_string_with_char_quote() {
        check_assist(
            replace_string_with_char,
            r#"
fn f() {
    find($0"'");
}
"#,
            r#"
fn f() {
    find('\'');
}
"#,
        )
    }

    #[test]
    fn replace_string_with_escaped_char_quote() {
        check_assist(
            replace_string_with_char,
            r#"
fn f() {
    find($0"\'");
}
"#,
            r#"
fn f() {
    find('\'');
}
"#,
        )
    }

    #[test]
    fn replace_string_with_char_quote_has_suffix() {
        check_assist(
            replace_string_with_char,
            r#"
fn f() {
    find($0"'"i32);
}
"#,
            r#"
fn f() {
    find('\''i32);
}
"#,
        )
    }

    #[test]
    fn replace_string_with_escaped_char_quote_has_suffix() {
        check_assist(
            replace_string_with_char,
            r#"
fn f() {
    find($0"\'"i32);
}
"#,
            r#"
fn f() {
    find('\''i32);
}
"#,
        )
    }

    #[test]
    fn replace_raw_string_with_char_quote() {
        check_assist(
            replace_string_with_char,
            r#"
fn f() {
    find($0r"'");
}
"#,
            r#"
fn f() {
    find('\'');
}
"#,
        )
    }

    #[test]
    fn replace_string_with_code_escaped_char_quote() {
        check_assist(
            replace_string_with_char,
            r#"
fn f() {
    find($0"\x27");
}
"#,
            r#"
fn f() {
    find('\x27');
}
"#,
        )
    }
}
