use std::borrow::Cow;

use ide_db::syntax_helpers::insert_whitespace_into_node::insert_ws_into;
use syntax::{ast, ast::IsString, AstNode, AstToken, TextRange, TextSize};

use crate::{utils::required_hashes, AssistContext, AssistId, AssistKind, Assists};

// Assist: make_raw_string
//
// Adds `r#` to a plain string literal.
//
// ```
// fn main() {
//     "Hello,$0 World!";
// }
// ```
// ->
// ```
// fn main() {
//     r#"Hello, World!"#;
// }
// ```
pub(crate) fn make_raw_string(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    // FIXME: This should support byte and c strings as well.
    let token = ctx.find_token_at_offset::<ast::String>()?;
    if token.is_raw() {
        return None;
    }
    let value = token.value()?;
    let target = token.syntax().text_range();
    acc.add(
        AssistId("make_raw_string", AssistKind::RefactorRewrite),
        "Rewrite as raw string",
        target,
        |edit| {
            let hashes = "#".repeat(required_hashes(&value).max(1));
            if matches!(value, Cow::Borrowed(_)) {
                // Avoid replacing the whole string to better position the cursor.
                edit.insert(token.syntax().text_range().start(), format!("r{hashes}"));
                edit.insert(token.syntax().text_range().end(), hashes);
            } else {
                edit.replace(token.syntax().text_range(), format!("r{hashes}\"{value}\"{hashes}"));
            }
        },
    )
}

// Assist: make_usual_string
//
// Turns a raw string into a plain string.
//
// ```
// fn main() {
//     r#"Hello,$0 "World!""#;
// }
// ```
// ->
// ```
// fn main() {
//     "Hello, \"World!\"";
// }
// ```
pub(crate) fn make_usual_string(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    let token = ctx.find_token_at_offset::<ast::String>()?;
    if !token.is_raw() {
        return None;
    }
    let value = token.value()?;
    let target = token.syntax().text_range();
    acc.add(
        AssistId("make_usual_string", AssistKind::RefactorRewrite),
        "Rewrite as regular string",
        target,
        |edit| {
            // parse inside string to escape `"`
            let escaped = value.escape_default().to_string();
            if let Some(offsets) = token.quote_offsets() {
                if token.text()[offsets.contents - token.syntax().text_range().start()] == escaped {
                    edit.replace(offsets.quotes.0, "\"");
                    edit.replace(offsets.quotes.1, "\"");
                    return;
                }
            }

            edit.replace(token.syntax().text_range(), format!("\"{escaped}\""));
        },
    )
}

// Assist: add_hash
//
// Adds a hash to a raw string literal.
//
// ```
// fn main() {
//     r#"Hello,$0 World!"#;
// }
// ```
// ->
// ```
// fn main() {
//     r##"Hello, World!"##;
// }
// ```
pub(crate) fn add_hash(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    let token = ctx.find_token_at_offset::<ast::String>()?;
    if !token.is_raw() {
        return None;
    }
    let text_range = token.syntax().text_range();
    let target = text_range;
    acc.add(AssistId("add_hash", AssistKind::Refactor), "Add #", target, |edit| {
        edit.insert(text_range.start() + TextSize::of('r'), "#");
        edit.insert(text_range.end(), "#");
    })
}

// Assist: remove_hash
//
// Removes a hash from a raw string literal.
//
// ```
// fn main() {
//     r#"Hello,$0 World!"#;
// }
// ```
// ->
// ```
// fn main() {
//     r"Hello, World!";
// }
// ```
pub(crate) fn remove_hash(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    let token = ctx.find_token_at_offset::<ast::String>()?;
    if !token.is_raw() {
        return None;
    }

    let text = token.text();
    if !text.starts_with("r#") && text.ends_with('#') {
        return None;
    }

    let existing_hashes = text.chars().skip(1).take_while(|&it| it == '#').count();

    let text_range = token.syntax().text_range();
    let internal_text = &text[token.text_range_between_quotes()? - text_range.start()];

    if existing_hashes == required_hashes(internal_text) {
        cov_mark::hit!(cant_remove_required_hash);
        return None;
    }

    acc.add(AssistId("remove_hash", AssistKind::RefactorRewrite), "Remove #", text_range, |edit| {
        edit.delete(TextRange::at(text_range.start() + TextSize::of('r'), TextSize::of('#')));
        edit.delete(TextRange::new(text_range.end() - TextSize::of('#'), text_range.end()));
    })
}

// Assist: inline_str_literal
//
// Inline const variable as static str literal.
//
// ```
// const STRING: &str = "Hello, World!";
//
// fn something() -> &'static str {
//     STR$0ING
// }
// ```
// ->
// ```
// const STRING: &str = "Hello, World!";
//
// fn something() -> &'static str {
//     "Hello, World!"
// }
// ```
pub(crate) fn inline_str_literal(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    let variable = ctx.find_node_at_offset::<ast::PathExpr>()?;

    if let hir::PathResolution::Def(hir::ModuleDef::Const(konst)) =
        ctx.sema.resolve_path(&variable.path()?)?
    {
        if !konst.ty(ctx.sema.db).as_reference()?.0.as_builtin()?.is_str() {
            return None;
        }

        // FIXME: Make sure it's not possible to eval during diagnostic error
        let value = match konst.value(ctx.sema.db)? {
            ast::Expr::Literal(lit) => lit.to_string(),
            ast::Expr::BlockExpr(_)
            | ast::Expr::IfExpr(_)
            | ast::Expr::MatchExpr(_)
            | ast::Expr::CallExpr(_) => match konst.render_eval(ctx.sema.db) {
                Ok(result) => result,
                Err(_) => return None,
            },
            ast::Expr::MacroExpr(makro) => {
                let makro_call = makro.syntax().children().find_map(ast::MacroCall::cast)?;
                let makro_hir = ctx.sema.resolve_macro_call(&makro_call)?;

                // This should not be necessary because of the `makro_call` check
                if !makro_hir.is_fn_like(ctx.sema.db) {
                    return None;
                }

                // FIXME: Make procedural/build-in macro tests
                insert_ws_into(ctx.sema.expand(&makro_call)?).to_string()
            }
            _ => return None,
        };

        let id = AssistId("inline_str_literal", AssistKind::RefactorInline);
        let label = "Inline as static `&str` literal";
        let target = variable.syntax().text_range();

        acc.add(id, label, target, |edit| {
            edit.replace(variable.syntax().text_range(), value);
        });
    }

    Some(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::{check_assist, check_assist_not_applicable, check_assist_target};

    #[test]
    fn make_raw_string_target() {
        check_assist_target(
            make_raw_string,
            r#"
            fn f() {
                let s = $0"random\nstring";
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
    let s = $0"random\nstring";
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
                format!($0"x = {}", 92)
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
    let s = $0"#random##\nstring";
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
    let s = $0"#random\"##\nstring";
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
                let s = $0"random string";
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
                let s = "foo$0
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
                let s = r#"bar$0
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
                let s = $0r"random string";
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
                let s = $0r"random string";
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
                let s = $0r#"random"string"#;
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
                let s = $0"random string";
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
                let s = $0r#"random string"#;
            }
            "##,
            r##"r#"random string"#"##,
        );
    }

    #[test]
    fn remove_hash_works() {
        check_assist(
            remove_hash,
            r##"fn f() { let s = $0r#"random string"#; }"##,
            r#"fn f() { let s = r"random string"; }"#,
        )
    }

    #[test]
    fn cant_remove_required_hash() {
        cov_mark::check!(cant_remove_required_hash);
        check_assist_not_applicable(
            remove_hash,
            r##"
            fn f() {
                let s = $0r#"random"str"ing"#;
            }
            "##,
        )
    }

    #[test]
    fn remove_more_hash_works() {
        check_assist(
            remove_hash,
            r###"
            fn f() {
                let s = $0r##"random string"##;
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
    fn remove_hash_doesnt_work() {
        check_assist_not_applicable(remove_hash, r#"fn f() { let s = $0"random string"; }"#);
    }

    #[test]
    fn remove_hash_no_hash_doesnt_work() {
        check_assist_not_applicable(remove_hash, r#"fn f() { let s = $0r"random string"; }"#);
    }

    #[test]
    fn make_usual_string_target() {
        check_assist_target(
            make_usual_string,
            r##"
            fn f() {
                let s = $0r#"random string"#;
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
                let s = $0r#"random string"#;
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
                let s = $0r#"random"str"ing"#;
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
                let s = $0r##"random string"##;
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
                let s = $0"random string";
            }
            "#,
        );
    }

    #[test]
    fn inline_expr_as_str_lit() {
        check_assist(
            inline_str_literal,
            r#"
            const STRING: &str = "Hello, World!";

            fn something() -> &'static str {
                STR$0ING
            }
            "#,
            r#"
            const STRING: &str = "Hello, World!";

            fn something() -> &'static str {
                "Hello, World!"
            }
            "#,
        );
    }

    #[test]
    fn inline_eval_const_block_expr_to_str_lit() {
        check_assist(
            inline_str_literal,
            r#"
            const STRING: &str = {
                let x = 9;
                if x + 10 == 21 {
                    "Hello, World!"
                } else {
                    "World, Hello!"
                }
            };

            fn something() -> &'static str {
                STR$0ING
            }
            "#,
            r#"
            const STRING: &str = {
                let x = 9;
                if x + 10 == 21 {
                    "Hello, World!"
                } else {
                    "World, Hello!"
                }
            };

            fn something() -> &'static str {
                "World, Hello!"
            }
            "#,
        );
    }

    #[test]
    fn inline_eval_const_block_macro_expr_to_str_lit() {
        check_assist(
            inline_str_literal,
            r#"
            macro_rules! co {() => {"World, Hello!"};}
            const STRING: &str = { co!() };

            fn something() -> &'static str {
                STR$0ING
            }
            "#,
            r#"
            macro_rules! co {() => {"World, Hello!"};}
            const STRING: &str = { co!() };

            fn something() -> &'static str {
                "World, Hello!"
            }
            "#,
        );
    }

    #[test]
    fn inline_eval_const_match_expr_to_str_lit() {
        check_assist(
            inline_str_literal,
            r#"
            const STRING: &str = match 9 + 10 {
                0..18 => "Hello, World!",
                _ => "World, Hello!"
            };

            fn something() -> &'static str {
                STR$0ING
            }
            "#,
            r#"
            const STRING: &str = match 9 + 10 {
                0..18 => "Hello, World!",
                _ => "World, Hello!"
            };

            fn something() -> &'static str {
                "World, Hello!"
            }
            "#,
        );
    }

    #[test]
    fn inline_eval_const_if_expr_to_str_lit() {
        check_assist(
            inline_str_literal,
            r#"
            const STRING: &str = if 1 + 2 == 4 {
                "Hello, World!"
            } else {
                "World, Hello!"
            }

            fn something() -> &'static str {
                STR$0ING
            }
            "#,
            r#"
            const STRING: &str = if 1 + 2 == 4 {
                "Hello, World!"
            } else {
                "World, Hello!"
            }

            fn something() -> &'static str {
                "World, Hello!"
            }
            "#,
        );
    }

    #[test]
    fn inline_eval_const_macro_expr_to_str_lit() {
        check_assist(
            inline_str_literal,
            r#"
            macro_rules! co {() => {"World, Hello!"};}
            const STRING: &str = co!();

            fn something() -> &'static str {
                STR$0ING
            }
            "#,
            r#"
            macro_rules! co {() => {"World, Hello!"};}
            const STRING: &str = co!();

            fn something() -> &'static str {
                "World, Hello!"
            }
            "#,
        );
    }

    #[test]
    fn inline_eval_const_call_expr_to_str_lit() {
        check_assist(
            inline_str_literal,
            r#"
            const fn const_call() -> &'static str {"World, Hello!"}
            const STRING: &str = const_call();

            fn something() -> &'static str {
                STR$0ING
            }
            "#,
            r#"
            const fn const_call() -> &'static str {"World, Hello!"}
            const STRING: &str = const_call();

            fn something() -> &'static str {
                "World, Hello!"
            }
            "#,
        );
    }

    #[test]
    fn inline_expr_as_str_lit_not_applicable() {
        check_assist_not_applicable(
            inline_str_literal,
            r#"
            const STRING: &str = "Hello, World!";

            fn something() -> &'static str {
                STRING $0
            }
            "#,
        );
    }

    #[test]
    fn inline_expr_as_str_lit_not_applicable_const() {
        check_assist_not_applicable(
            inline_str_literal,
            r#"
            const STR$0ING: &str = "Hello, World!";

            fn something() -> &'static str {
                STRING
            }
            "#,
        );
    }
}
