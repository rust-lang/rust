use hir::db::HirDatabase;
use ra_syntax::{ast::AstNode, ast::Literal, TextRange, TextUnit};

use crate::{Assist, AssistCtx, AssistId};

pub(crate) fn make_raw_string(mut ctx: AssistCtx<impl HirDatabase>) -> Option<Assist> {
    let literal = ctx.node_at_offset::<Literal>()?;
    if literal.token().kind() == ra_syntax::SyntaxKind::STRING {
        ctx.add_action(AssistId("make_raw_string"), "make raw string", |edit| {
            edit.target(literal.syntax().text_range());
            edit.insert(literal.syntax().text_range().start(), "r");
        });
        ctx.build()
    } else {
        None
    }
}

pub(crate) fn make_usual_string(mut ctx: AssistCtx<impl HirDatabase>) -> Option<Assist> {
    let literal = ctx.node_at_offset::<Literal>()?;
    if literal.token().kind() == ra_syntax::SyntaxKind::RAW_STRING {
        ctx.add_action(AssistId("make_usual_string"), "make usual string", |edit| {
            let text = literal.syntax().text();
            let usual_start_pos = text.find_char('"').unwrap(); // we have a RAW_STRING
            let end = literal.syntax().text_range().end();
            dbg!(&end);
            let mut i = 0;
            let mut pos = 0;
            let mut c = text.char_at(end - TextUnit::from(i));
            while c != Some('"') {
                if c != None {
                    pos += 1;
                }
                i += 1;
                c = text.char_at(end - TextUnit::from(i));
            }

            edit.target(literal.syntax().text_range());
            edit.delete(TextRange::from_to(
                literal.syntax().text_range().start(),
                literal.syntax().text_range().start() + usual_start_pos,
            ));
            edit.delete(TextRange::from_to(
                literal.syntax().text_range().end() - TextUnit::from(pos),
                literal.syntax().text_range().end(),
            ));
        });
        ctx.build()
    } else {
        None
    }
}

pub(crate) fn add_hash(mut ctx: AssistCtx<impl HirDatabase>) -> Option<Assist> {
    let literal = ctx.node_at_offset::<Literal>()?;
    if literal.token().kind() == ra_syntax::SyntaxKind::RAW_STRING {
        ctx.add_action(AssistId("add_hash"), "add hash to raw string", |edit| {
            edit.target(literal.syntax().text_range());
            edit.insert(literal.syntax().text_range().start() + TextUnit::from(1), "#");
            edit.insert(literal.syntax().text_range().end(), "#");
        });
        ctx.build()
    } else {
        None
    }
}

pub(crate) fn remove_hash(mut ctx: AssistCtx<impl HirDatabase>) -> Option<Assist> {
    let literal = ctx.node_at_offset::<Literal>()?;
    if literal.token().kind() == ra_syntax::SyntaxKind::RAW_STRING {
        if !literal.syntax().text().contains_char('#') {
            return None;
        }
        ctx.add_action(AssistId("remove_hash"), "remove hash from raw string", |edit| {
            edit.target(literal.syntax().text_range());
            edit.delete(TextRange::from_to(
                literal.syntax().text_range().start() + TextUnit::from(1),
                literal.syntax().text_range().start() + TextUnit::from(2),
            ));
            edit.delete(TextRange::from_to(
                literal.syntax().text_range().end() - TextUnit::from(1),
                literal.syntax().text_range().end(),
            ));
        });
        ctx.build()
    } else {
        None
    }
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
                let s = <|>"random string";
            }
            "#,
            r#""random string""#,
        );
    }

    #[test]
    fn make_raw_string_works() {
        check_assist(
            make_raw_string,
            r#"
            fn f() {
                let s = <|>"random string";
            }
            "#,
            r#"
            fn f() {
                let s = <|>r"random string";
            }
            "#,
        )
    }

    #[test]
    fn make_raw_string_not_works() {
        check_assist_not_applicable(
            make_raw_string,
            r#"
            fn f() {
                let s = <|>r"random string";
            }
            "#,
        );
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
                let s = <|>r#"random string"#;
            }
            "##,
            r###"
            fn f() {
                let s = <|>r##"random string"##;
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
}
