use ra_syntax::{
    ast::{self, HasStringValue},
    AstToken,
    SyntaxKind::{RAW_STRING, STRING},
    TextSize,
};

use crate::{AssistContext, AssistId, AssistKind, Assists};

// Assist: make_raw_string
//
// Adds `r#` to a plain string literal.
//
// ```
// fn main() {
//     "Hello,<|> World!";
// }
// ```
// ->
// ```
// fn main() {
//     r#"Hello, World!"#;
// }
// ```
pub(crate) fn make_raw_string(acc: &mut Assists, ctx: &AssistContext) -> Option<()> {
    let token = ctx.find_token_at_offset(STRING).and_then(ast::String::cast)?;
    let value = token.value()?;
    let target = token.syntax().text_range();
    acc.add(
        AssistId("make_raw_string", AssistKind::RefactorRewrite),
        "Rewrite as raw string",
        target,
        |edit| {
            let max_hash_streak = count_hashes(&value);
            let mut hashes = String::with_capacity(max_hash_streak + 1);
            for _ in 0..hashes.capacity() {
                hashes.push('#');
            }
            edit.replace(
                token.syntax().text_range(),
                format!("r{}\"{}\"{}", hashes, value, hashes),
            );
        },
    )
}

// Assist: make_usual_string
//
// Turns a raw string into a plain string.
//
// ```
// fn main() {
//     r#"Hello,<|> "World!""#;
// }
// ```
// ->
// ```
// fn main() {
//     "Hello, \"World!\"";
// }
// ```
pub(crate) fn make_usual_string(acc: &mut Assists, ctx: &AssistContext) -> Option<()> {
    let token = ctx.find_token_at_offset(RAW_STRING).and_then(ast::RawString::cast)?;
    let value = token.value()?;
    let target = token.syntax().text_range();
    acc.add(
        AssistId("make_usual_string", AssistKind::RefactorRewrite),
        "Rewrite as regular string",
        target,
        |edit| {
            // parse inside string to escape `"`
            let escaped = value.escape_default().to_string();
            edit.replace(token.syntax().text_range(), format!("\"{}\"", escaped));
        },
    )
}

// Assist: add_hash
//
// Adds a hash to a raw string literal.
//
// ```
// fn main() {
//     r#"Hello,<|> World!"#;
// }
// ```
// ->
// ```
// fn main() {
//     r##"Hello, World!"##;
// }
// ```
pub(crate) fn add_hash(acc: &mut Assists, ctx: &AssistContext) -> Option<()> {
    let token = ctx.find_token_at_offset(RAW_STRING)?;
    let target = token.text_range();
    acc.add(AssistId("add_hash", AssistKind::Refactor), "Add # to raw string", target, |edit| {
        edit.insert(token.text_range().start() + TextSize::of('r'), "#");
        edit.insert(token.text_range().end(), "#");
    })
}

// Assist: remove_hash
//
// Removes a hash from a raw string literal.
//
// ```
// fn main() {
//     r#"Hello,<|> World!"#;
// }
// ```
// ->
// ```
// fn main() {
//     r"Hello, World!";
// }
// ```
pub(crate) fn remove_hash(acc: &mut Assists, ctx: &AssistContext) -> Option<()> {
    let token = ctx.find_token_at_offset(RAW_STRING)?;
    let text = token.text().as_str();
    if text.starts_with("r\"") {
        // no hash to remove
        return None;
    }
    let target = token.text_range();
    acc.add(
        AssistId("remove_hash", AssistKind::RefactorRewrite),
        "Remove hash from raw string",
        target,
        |edit| {
            let result = &text[2..text.len() - 1];
            let result = if result.starts_with('\"') {
                // FIXME: this logic is wrong, not only the last has has to handled specially
                // no more hash, escape
                let internal_str = &result[1..result.len() - 1];
                format!("\"{}\"", internal_str.escape_default().to_string())
            } else {
                result.to_owned()
            };
            edit.replace(token.text_range(), format!("r{}", result));
        },
    )
}

fn count_hashes(s: &str) -> usize {
    let mut max_hash_streak = 0usize;
    for idx in s.match_indices("\"#").map(|(i, _)| i) {
        let (_, sub) = s.split_at(idx + 1);
        let nb_hash = sub.chars().take_while(|c| *c == '#').count();
        if nb_hash > max_hash_streak {
            max_hash_streak = nb_hash;
        }
    }
    max_hash_streak
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::tests::{check_assist, check_assist_not_applicable, check_assist_target};

    #[test]
    fn make_raw_string_target() {
        check_assist_target(
            make_raw_string,
            r#"
            fn f() {
                let s = <|>"random\nstring";
            }
            "#,
            r#""random\nstring""#,
        );
    }

    #[test]
    fn make_raw_string_works() {
        check_assist(
            make_raw_string,
            r#"
fn f() {
    let s = <|>"random\nstring";
}
"#,
            r##"
fn f() {
    let s = r#"random
string"#;
}
"##,
        )
    }

    #[test]
    fn make_raw_string_works_inside_macros() {
        check_assist(
            make_raw_string,
            r#"
            fn f() {
                format!(<|>"x = {}", 92)
            }
            "#,
            r##"
            fn f() {
                format!(r#"x = {}"#, 92)
            }
            "##,
        )
    }

    #[test]
    fn make_raw_string_hashes_inside_works() {
        check_assist(
            make_raw_string,
            r###"
fn f() {
    let s = <|>"#random##\nstring";
}
"###,
            r####"
fn f() {
    let s = r#"#random##
string"#;
}
"####,
        )
    }

    #[test]
    fn make_raw_string_closing_hashes_inside_works() {
        check_assist(
            make_raw_string,
            r###"
fn f() {
    let s = <|>"#random\"##\nstring";
}
"###,
            r####"
fn f() {
    let s = r###"#random"##
string"###;
}
"####,
        )
    }

    #[test]
    fn make_raw_string_nothing_to_unescape_works() {
        check_assist(
            make_raw_string,
            r#"
            fn f() {
                let s = <|>"random string";
            }
            "#,
            r##"
            fn f() {
                let s = r#"random string"#;
            }
            "##,
        )
    }

    #[test]
    fn make_raw_string_not_works_on_partial_string() {
        check_assist_not_applicable(
            make_raw_string,
            r#"
            fn f() {
                let s = "foo<|>
            }
            "#,
        )
    }

    #[test]
    fn make_usual_string_not_works_on_partial_string() {
        check_assist_not_applicable(
            make_usual_string,
            r#"
            fn main() {
                let s = r#"bar<|>
            }
            "#,
        )
    }

    #[test]
    fn add_hash_target() {
        check_assist_target(
            add_hash,
            r#"
            fn f() {
                let s = <|>r"random string";
            }
            "#,
            r#"r"random string""#,
        );
    }

    #[test]
    fn add_hash_works() {
        check_assist(
            add_hash,
            r#"
            fn f() {
                let s = <|>r"random string";
            }
            "#,
            r##"
            fn f() {
                let s = r#"random string"#;
            }
            "##,
        )
    }

    #[test]
    fn add_more_hash_works() {
        check_assist(
            add_hash,
            r##"
            fn f() {
                let s = <|>r#"random"string"#;
            }
            "##,
            r###"
            fn f() {
                let s = r##"random"string"##;
            }
            "###,
        )
    }

    #[test]
    fn add_hash_not_works() {
        check_assist_not_applicable(
            add_hash,
            r#"
            fn f() {
                let s = <|>"random string";
            }
            "#,
        );
    }

    #[test]
    fn remove_hash_target() {
        check_assist_target(
            remove_hash,
            r##"
            fn f() {
                let s = <|>r#"random string"#;
            }
            "##,
            r##"r#"random string"#"##,
        );
    }

    #[test]
    fn remove_hash_works() {
        check_assist(
            remove_hash,
            r##"
            fn f() {
                let s = <|>r#"random string"#;
            }
            "##,
            r#"
            fn f() {
                let s = r"random string";
            }
            "#,
        )
    }

    #[test]
    fn remove_hash_with_quote_works() {
        check_assist(
            remove_hash,
            r##"
            fn f() {
                let s = <|>r#"random"str"ing"#;
            }
            "##,
            r#"
            fn f() {
                let s = r"random\"str\"ing";
            }
            "#,
        )
    }

    #[test]
    fn remove_more_hash_works() {
        check_assist(
            remove_hash,
            r###"
            fn f() {
                let s = <|>r##"random string"##;
            }
            "###,
            r##"
            fn f() {
                let s = r#"random string"#;
            }
            "##,
        )
    }

    #[test]
    fn remove_hash_not_works() {
        check_assist_not_applicable(
            remove_hash,
            r#"
            fn f() {
                let s = <|>"random string";
            }
            "#,
        );
    }

    #[test]
    fn remove_hash_no_hash_not_works() {
        check_assist_not_applicable(
            remove_hash,
            r#"
            fn f() {
                let s = <|>r"random string";
            }
            "#,
        );
    }

    #[test]
    fn make_usual_string_target() {
        check_assist_target(
            make_usual_string,
            r##"
            fn f() {
                let s = <|>r#"random string"#;
            }
            "##,
            r##"r#"random string"#"##,
        );
    }

    #[test]
    fn make_usual_string_works() {
        check_assist(
            make_usual_string,
            r##"
            fn f() {
                let s = <|>r#"random string"#;
            }
            "##,
            r#"
            fn f() {
                let s = "random string";
            }
            "#,
        )
    }

    #[test]
    fn make_usual_string_with_quote_works() {
        check_assist(
            make_usual_string,
            r##"
            fn f() {
                let s = <|>r#"random"str"ing"#;
            }
            "##,
            r#"
            fn f() {
                let s = "random\"str\"ing";
            }
            "#,
        )
    }

    #[test]
    fn make_usual_string_more_hash_works() {
        check_assist(
            make_usual_string,
            r###"
            fn f() {
                let s = <|>r##"random string"##;
            }
            "###,
            r##"
            fn f() {
                let s = "random string";
            }
            "##,
        )
    }

    #[test]
    fn make_usual_string_not_works() {
        check_assist_not_applicable(
            make_usual_string,
            r#"
            fn f() {
                let s = <|>"random string";
            }
            "#,
        );
    }

    #[test]
    fn count_hashes_test() {
        assert_eq!(0, count_hashes("abc"));
        assert_eq!(0, count_hashes("###"));
        assert_eq!(1, count_hashes("\"#abc"));
        assert_eq!(0, count_hashes("#abc"));
        assert_eq!(2, count_hashes("#ab\"##c"));
        assert_eq!(4, count_hashes("#ab\"##\"####c"));
    }
}
