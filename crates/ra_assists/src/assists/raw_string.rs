//! FIXME: write short doc here

use hir::db::HirDatabase;
use ra_syntax::{
    SyntaxKind::{RAW_STRING, STRING},
    SyntaxToken, TextRange, TextUnit,
};
use rustc_lexer;

use crate::{Assist, AssistCtx, AssistId};

pub(crate) fn make_raw_string(mut ctx: AssistCtx<impl HirDatabase>) -> Option<Assist> {
    let token = ctx.token_at_offset().right_biased()?;
    if token.kind() != STRING {
        return None;
    }
    let text = token.text().as_str();
    let usual_string_range = find_usual_string_range(text)?;
    let start_of_inside = usual_string_range.start().to_usize() + 1;
    let end_of_inside = usual_string_range.end().to_usize();
    let inside_str = &text[start_of_inside..end_of_inside];
    let mut unescaped = String::with_capacity(inside_str.len());
    let mut error = Ok(());
    rustc_lexer::unescape::unescape_str(
        inside_str,
        &mut |_, unescaped_char| match unescaped_char {
            Ok(c) => unescaped.push(c),
            Err(_) => error = Err(()),
        },
    );
    if error.is_err() {
        return None;
    }
    ctx.add_action(AssistId("make_raw_string"), "make raw string", |edit| {
        edit.target(token.text_range());
        let max_hash_streak = count_hashes(&unescaped);
        let mut hashes = String::with_capacity(max_hash_streak + 1);
        for _ in 0..hashes.capacity() {
            hashes.push('#');
        }
        edit.replace(token.text_range(), format!("r{}\"{}\"{}", hashes, unescaped, hashes));
    });
    ctx.build()
}

pub(crate) fn make_usual_string(mut ctx: AssistCtx<impl HirDatabase>) -> Option<Assist> {
    let token = raw_string_token(&ctx)?;
    let text = token.text().as_str();
    let usual_string_range = find_usual_string_range(text)?;
    ctx.add_action(AssistId("make_usual_string"), "make usual string", |edit| {
        edit.target(token.text_range());
        // parse inside string to escape `"`
        let start_of_inside = usual_string_range.start().to_usize() + 1;
        let end_of_inside = usual_string_range.end().to_usize();
        let inside_str = &text[start_of_inside..end_of_inside];
        let escaped = inside_str.escape_default().to_string();
        edit.replace(token.text_range(), format!("\"{}\"", escaped));
    });
    ctx.build()
}

pub(crate) fn add_hash(mut ctx: AssistCtx<impl HirDatabase>) -> Option<Assist> {
    let token = raw_string_token(&ctx)?;
    ctx.add_action(AssistId("add_hash"), "add hash to raw string", |edit| {
        edit.target(token.text_range());
        edit.insert(token.text_range().start() + TextUnit::of_char('r'), "#");
        edit.insert(token.text_range().end(), "#");
    });
    ctx.build()
}

pub(crate) fn remove_hash(mut ctx: AssistCtx<impl HirDatabase>) -> Option<Assist> {
    let token = raw_string_token(&ctx)?;
    let text = token.text().as_str();
    if text.starts_with("r\"") {
        // no hash to remove
        return None;
    }
    ctx.add_action(AssistId("remove_hash"), "remove hash from raw string", |edit| {
        edit.target(token.text_range());
        let result = &text[2..text.len() - 1];
        let result = if result.starts_with("\"") {
            // no more hash, escape
            let internal_str = &result[1..result.len() - 1];
            format!("\"{}\"", internal_str.escape_default().to_string())
        } else {
            result.to_owned()
        };
        edit.replace(token.text_range(), format!("r{}", result));
    });
    ctx.build()
}

fn raw_string_token(ctx: &AssistCtx<impl HirDatabase>) -> Option<SyntaxToken> {
    ctx.token_at_offset().right_biased().filter(|it| it.kind() == RAW_STRING)
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

fn find_usual_string_range(s: &str) -> Option<TextRange> {
    Some(TextRange::from_to(
        TextUnit::from(s.find('"')? as u32),
        TextUnit::from(s.rfind('"')? as u32),
    ))
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::helpers::{check_assist, check_assist_not_applicable, check_assist_target};

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
                let s = <|>r#"random
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
                format!(<|>r#"x = {}"#, 92)
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
                let s = <|>r#"#random##
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
                let s = <|>r###"#random"##
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
                let s = <|>r#"random string"#;
            }
            "##,
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
                let s = <|>r#"random string"#;
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
                let s = <|>r##"random"string"##;
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
                let s = <|>r"random string";
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
                let s = <|>r"random\"str\"ing";
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
                let s = <|>r#"random string"#;
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
                let s = <|>"random string";
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
                let s = <|>"random\"str\"ing";
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
                let s = <|>"random string";
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
