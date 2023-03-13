use itertools::Itertools;
use syntax::{
    ast::{self, make, AstNode, AstToken},
    match_ast, ted, NodeOrToken, SyntaxElement, TextRange, TextSize, T,
};

use crate::{AssistContext, AssistId, AssistKind, Assists};

// Assist: remove_dbg
//
// Removes `dbg!()` macro call.
//
// ```
// fn main() {
//     let x = $0dbg!(42 * dbg!(4 + 2));$0
// }
// ```
// ->
// ```
// fn main() {
//     let x = 42 * (4 + 2);
// }
// ```
pub(crate) fn remove_dbg(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    let macro_calls = if ctx.has_empty_selection() {
        vec![ctx.find_node_at_offset::<ast::MacroExpr>()?]
    } else {
        ctx.covering_element()
            .as_node()?
            .descendants()
            .filter(|node| ctx.selection_trimmed().contains_range(node.text_range()))
            // When the selection exactly covers the macro call to be removed, `covering_element()`
            // returns `ast::MacroCall` instead of its parent `ast::MacroExpr` that we want. So
            // first try finding `ast::MacroCall`s and then retrieve their parent.
            .filter_map(ast::MacroCall::cast)
            .filter_map(|it| it.syntax().parent().and_then(ast::MacroExpr::cast))
            .collect()
    };

    let replacements =
        macro_calls.into_iter().filter_map(compute_dbg_replacement).collect::<Vec<_>>();
    if replacements.is_empty() {
        return None;
    }

    acc.add(
        AssistId("remove_dbg", AssistKind::Refactor),
        "Remove dbg!()",
        ctx.selection_trimmed(),
        |builder| {
            for (range, expr) in replacements {
                if let Some(expr) = expr {
                    builder.replace(range, expr.to_string());
                } else {
                    builder.delete(range);
                }
            }
        },
    )
}

/// Returns `None` when either
/// - macro call is not `dbg!()`
/// - any node inside `dbg!()` could not be parsed as an expression
/// - (`macro_expr` has no parent - is that possible?)
///
/// Returns `Some(_, None)` when the macro call should just be removed.
fn compute_dbg_replacement(macro_expr: ast::MacroExpr) -> Option<(TextRange, Option<ast::Expr>)> {
    let macro_call = macro_expr.macro_call()?;
    let tt = macro_call.token_tree()?;
    let r_delim = NodeOrToken::Token(tt.right_delimiter_token()?);
    if macro_call.path()?.segment()?.name_ref()?.text() != "dbg"
        || macro_call.excl_token().is_none()
    {
        return None;
    }

    let mac_input = tt.syntax().children_with_tokens().skip(1).take_while(|it| *it != r_delim);
    let input_expressions = mac_input.group_by(|tok| tok.kind() == T![,]);
    let input_expressions = input_expressions
        .into_iter()
        .filter_map(|(is_sep, group)| (!is_sep).then_some(group))
        .map(|mut tokens| syntax::hacks::parse_expr_from_str(&tokens.join("")))
        .collect::<Option<Vec<ast::Expr>>>()?;

    let parent = macro_expr.syntax().parent()?;
    Some(match &*input_expressions {
        // dbg!()
        [] => {
            match_ast! {
                match parent {
                    ast::StmtList(_) => {
                        let range = macro_expr.syntax().text_range();
                        let range = match whitespace_start(macro_expr.syntax().prev_sibling_or_token()) {
                            Some(start) => range.cover_offset(start),
                            None => range,
                        };
                        (range, None)
                    },
                    ast::ExprStmt(it) => {
                        let range = it.syntax().text_range();
                        let range = match whitespace_start(it.syntax().prev_sibling_or_token()) {
                            Some(start) => range.cover_offset(start),
                            None => range,
                        };
                        (range, None)
                    },
                    _ => (macro_call.syntax().text_range(), Some(make::expr_unit())),
                }
            }
        }
        // dbg!(expr0)
        [expr] => {
            // dbg!(expr, &parent);
            let wrap = match ast::Expr::cast(parent) {
                Some(parent) => match (expr, parent) {
                    (ast::Expr::CastExpr(_), ast::Expr::CastExpr(_)) => false,
                    (
                        ast::Expr::BoxExpr(_)
                        | ast::Expr::PrefixExpr(_)
                        | ast::Expr::RefExpr(_)
                        | ast::Expr::MacroExpr(_),
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
                        ast::Expr::BinExpr(_)
                        | ast::Expr::CastExpr(_)
                        | ast::Expr::RangeExpr(_)
                        | ast::Expr::MacroExpr(_),
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
            let expr = replace_nested_dbgs(expr.clone());
            let expr = if wrap { make::expr_paren(expr) } else { expr.clone_subtree() };
            (macro_call.syntax().text_range(), Some(expr))
        }
        // dbg!(expr0, expr1, ...)
        exprs => {
            let exprs = exprs.iter().cloned().map(replace_nested_dbgs);
            let expr = make::expr_tuple(exprs);
            (macro_call.syntax().text_range(), Some(expr))
        }
    })
}

fn replace_nested_dbgs(expanded: ast::Expr) -> ast::Expr {
    if let ast::Expr::MacroExpr(mac) = &expanded {
        // Special-case when `expanded` itself is `dbg!()` since we cannot replace the whole tree
        // with `ted`. It should be fairly rare as it means the user wrote `dbg!(dbg!(..))` but you
        // never know how code ends up being!
        let replaced = if let Some((_, expr_opt)) = compute_dbg_replacement(mac.clone()) {
            match expr_opt {
                Some(expr) => expr,
                None => {
                    stdx::never!("dbg! inside dbg! should not be just removed");
                    expanded
                }
            }
        } else {
            expanded
        };

        return replaced;
    }

    let expanded = expanded.clone_for_update();

    // We need to collect to avoid mutation during traversal.
    let macro_exprs: Vec<_> =
        expanded.syntax().descendants().filter_map(ast::MacroExpr::cast).collect();

    for mac in macro_exprs {
        let expr_opt = match compute_dbg_replacement(mac.clone()) {
            Some((_, expr)) => expr,
            None => continue,
        };

        if let Some(expr) = expr_opt {
            ted::replace(mac.syntax(), expr.syntax().clone_for_update());
        } else {
            ted::remove(mac.syntax());
        }
    }

    expanded
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
            &format!("fn main() {{\n{ra_fixture_before}\n}}"),
            &format!("fn main() {{\n{ra_fixture_after}\n}}"),
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
        // https://github.com/rust-lang/rust-analyzer/issues/5129#issuecomment-651399779
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

    #[test]
    fn test_range() {
        check(
            r#"
fn f() {
    dbg!(0) + $0dbg!(1);
    dbg!(())$0
}
"#,
            r#"
fn f() {
    dbg!(0) + 1;
    ()
}
"#,
        );
    }

    #[test]
    fn test_range_partial() {
        check_assist_not_applicable(remove_dbg, r#"$0dbg$0!(0)"#);
        check_assist_not_applicable(remove_dbg, r#"$0dbg!(0$0)"#);
    }

    #[test]
    fn test_nested_dbg() {
        check(
            r#"$0let x = dbg!(dbg!(dbg!(dbg!(0 + 1)) * 2) + dbg!(3));$0"#,
            r#"let x = ((0 + 1) * 2) + 3;"#,
        );
        check(r#"$0dbg!(10, dbg!(), dbg!(20, 30))$0"#, r#"(10, (), (20, 30))"#);
    }

    #[test]
    fn test_multiple_nested_dbg() {
        check(
            r#"
fn f() {
    $0dbg!();
    let x = dbg!(dbg!(dbg!(0 + 1)) + 2) + dbg!(3);
    dbg!(10, dbg!(), dbg!(20, 30));$0
}
"#,
            r#"
fn f() {
    let x = ((0 + 1) + 2) + 3;
    (10, (), (20, 30));
}
"#,
        );
    }
}
