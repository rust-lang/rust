use itertools::Itertools;
use syntax::{
    ast::{self, AstNode, AstToken},
    match_ast, NodeOrToken, SyntaxElement, TextSize, T,
};

use crate::{AssistContext, AssistId, AssistKind, Assists};

// Assist: remove_dbg
//
// Removes `dbg!()` macro call.
//
// ```
// fn main() {
//     $0dbg!(92);
// }
// ```
// ->
// ```
// fn main() {
//     92;
// }
// ```
pub(crate) fn remove_dbg(acc: &mut Assists, ctx: &AssistContext) -> Option<()> {
    let macro_call = ctx.find_node_at_offset::<ast::MacroCall>()?;
    let tt = macro_call.token_tree()?;
    let r_delim = NodeOrToken::Token(tt.right_delimiter_token()?);
    if macro_call.path()?.segment()?.name_ref()?.text() != "dbg"
        || macro_call.excl_token().is_none()
    {
        return None;
    }

    let mac_input = tt.syntax().children_with_tokens().skip(1).take_while(|it| *it != r_delim);
    let input_expressions = mac_input.into_iter().group_by(|tok| tok.kind() == T![,]);
    let input_expressions = input_expressions
        .into_iter()
        .filter_map(|(is_sep, group)| (!is_sep).then(|| group))
        .map(|mut tokens| syntax::hacks::parse_expr_from_str(&tokens.join("")))
        .collect::<Option<Vec<ast::Expr>>>()?;

    let parent = macro_call.syntax().parent()?;
    let (range, text) = match &*input_expressions {
        // dbg!()
        [] => {
            match_ast! {
                match parent {
                    ast::StmtList(__) => {
                        let range = macro_call.syntax().text_range();
                        let range = match whitespace_start(macro_call.syntax().prev_sibling_or_token()) {
                            Some(start) => range.cover_offset(start),
                            None => range,
                        };
                        (range, String::new())
                    },
                    ast::ExprStmt(it) => {
                        let range = it.syntax().text_range();
                        let range = match whitespace_start(it.syntax().prev_sibling_or_token()) {
                            Some(start) => range.cover_offset(start),
                            None => range,
                        };
                        (range, String::new())
                    },
                    _ => (macro_call.syntax().text_range(), "()".to_owned())
                }
            }
        }
        // dbg!(expr0)
        [expr] => {
            let wrap = match ast::Expr::cast(parent) {
                Some(parent) => match (expr, parent) {
                    (ast::Expr::CastExpr(_), ast::Expr::CastExpr(_)) => false,
                    (
                        ast::Expr::BoxExpr(_) | ast::Expr::PrefixExpr(_) | ast::Expr::RefExpr(_),
                        ast::Expr::AwaitExpr(_)
                        | ast::Expr::CallExpr(_)
                        | ast::Expr::CastExpr(_)
                        | ast::Expr::FieldExpr(_)
                        | ast::Expr::IndexExpr(_)
                        | ast::Expr::MethodCallExpr(_)
                        | ast::Expr::RangeExpr(_)
                        | ast::Expr::TryExpr(_),
                    ) => true,
                    (
                        ast::Expr::BinExpr(_) | ast::Expr::CastExpr(_) | ast::Expr::RangeExpr(_),
                        ast::Expr::AwaitExpr(_)
                        | ast::Expr::BinExpr(_)
                        | ast::Expr::CallExpr(_)
                        | ast::Expr::CastExpr(_)
                        | ast::Expr::FieldExpr(_)
                        | ast::Expr::IndexExpr(_)
                        | ast::Expr::MethodCallExpr(_)
                        | ast::Expr::PrefixExpr(_)
                        | ast::Expr::RangeExpr(_)
                        | ast::Expr::RefExpr(_)
                        | ast::Expr::TryExpr(_),
                    ) => true,
                    _ => false,
                },
                None => false,
            };
            (
                macro_call.syntax().text_range(),
                if wrap { format!("({})", expr) } else { expr.to_string() },
            )
        }
        // dbg!(expr0, expr1, ...)
        exprs => (macro_call.syntax().text_range(), format!("({})", exprs.iter().format(", "))),
    };

    acc.add(AssistId("remove_dbg", AssistKind::Refactor), "Remove dbg!()", range, |builder| {
        builder.replace(range, text);
    })
}

fn whitespace_start(it: Option<SyntaxElement>) -> Option<TextSize> {
    Some(it?.into_token().and_then(ast::Whitespace::cast)?.syntax().text_range().start())
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_assist, check_assist_not_applicable};

    use super::*;

    fn check(ra_fixture_before: &str, ra_fixture_after: &str) {
        check_assist(
            remove_dbg,
            &format!("fn main() {{\n{}\n}}", ra_fixture_before),
            &format!("fn main() {{\n{}\n}}", ra_fixture_after),
        );
    }

    #[test]
    fn test_remove_dbg() {
        check("$0dbg!(1 + 1)", "1 + 1");
        check("dbg!$0(1 + 1)", "1 + 1");
        check("dbg!(1 $0+ 1)", "1 + 1");
        check("dbg![$01 + 1]", "1 + 1");
        check("dbg!{$01 + 1}", "1 + 1");
    }

    #[test]
    fn test_remove_dbg_not_applicable() {
        check_assist_not_applicable(remove_dbg, "fn main() {$0vec![1, 2, 3]}");
        check_assist_not_applicable(remove_dbg, "fn main() {$0dbg(5, 6, 7)}");
        check_assist_not_applicable(remove_dbg, "fn main() {$0dbg!(5, 6, 7}");
    }

    #[test]
    fn test_remove_dbg_keep_semicolon_in_let() {
        // https://github.com/rust-analyzer/rust-analyzer/issues/5129#issuecomment-651399779
        check(
            r#"let res = $0dbg!(1 * 20); // needless comment"#,
            r#"let res = 1 * 20; // needless comment"#,
        );
        check(r#"let res = $0dbg!(); // needless comment"#, r#"let res = (); // needless comment"#);
        check(
            r#"let res = $0dbg!(1, 2); // needless comment"#,
            r#"let res = (1, 2); // needless comment"#,
        );
    }

    #[test]
    fn test_remove_dbg_cast_cast() {
        check(r#"let res = $0dbg!(x as u32) as u32;"#, r#"let res = x as u32 as u32;"#);
    }

    #[test]
    fn test_remove_dbg_prefix() {
        check(r#"let res = $0dbg!(&result).foo();"#, r#"let res = (&result).foo();"#);
        check(r#"let res = &$0dbg!(&result);"#, r#"let res = &&result;"#);
        check(r#"let res = $0dbg!(!result) && true;"#, r#"let res = !result && true;"#);
    }

    #[test]
    fn test_remove_dbg_post_expr() {
        check(r#"let res = $0dbg!(fut.await).foo();"#, r#"let res = fut.await.foo();"#);
        check(r#"let res = $0dbg!(result?).foo();"#, r#"let res = result?.foo();"#);
        check(r#"let res = $0dbg!(foo as u32).foo();"#, r#"let res = (foo as u32).foo();"#);
        check(r#"let res = $0dbg!(array[3]).foo();"#, r#"let res = array[3].foo();"#);
        check(r#"let res = $0dbg!(tuple.3).foo();"#, r#"let res = tuple.3.foo();"#);
    }

    #[test]
    fn test_remove_dbg_range_expr() {
        check(r#"let res = $0dbg!(foo..bar).foo();"#, r#"let res = (foo..bar).foo();"#);
        check(r#"let res = $0dbg!(foo..=bar).foo();"#, r#"let res = (foo..=bar).foo();"#);
    }

    #[test]
    fn test_remove_empty_dbg() {
        check_assist(remove_dbg, r#"fn foo() { $0dbg!(); }"#, r#"fn foo() { }"#);
        check_assist(
            remove_dbg,
            r#"
fn foo() {
    $0dbg!();
}
"#,
            r#"
fn foo() {
}
"#,
        );
        check_assist(
            remove_dbg,
            r#"
fn foo() {
    let test = $0dbg!();
}"#,
            r#"
fn foo() {
    let test = ();
}"#,
        );
        check_assist(
            remove_dbg,
            r#"
fn foo() {
    let t = {
        println!("Hello, world");
        $0dbg!()
    };
}"#,
            r#"
fn foo() {
    let t = {
        println!("Hello, world");
    };
}"#,
        );
    }

    #[test]
    fn test_remove_multi_dbg() {
        check(r#"$0dbg!(0, 1)"#, r#"(0, 1)"#);
        check(r#"$0dbg!(0, (1, 2))"#, r#"(0, (1, 2))"#);
    }
}
