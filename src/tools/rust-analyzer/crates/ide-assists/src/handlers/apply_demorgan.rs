use std::collections::VecDeque;

use ide_db::{
    assists::GroupLabel,
    famous_defs::FamousDefs,
    source_change::SourceChangeBuilder,
    syntax_helpers::node_ext::{for_each_tail_expr, walk_expr},
};
use syntax::{
    ast::{self, make, AstNode, Expr::BinExpr, HasArgList},
    ted, SyntaxKind, T,
};

use crate::{utils::invert_boolean_expression, AssistContext, AssistId, AssistKind, Assists};

// Assist: apply_demorgan
//
// Apply https://en.wikipedia.org/wiki/De_Morgan%27s_laws[De Morgan's law].
// This transforms expressions of the form `!l || !r` into `!(l && r)`.
// This also works with `&&`. This assist can only be applied with the cursor
// on either `||` or `&&`.
//
// ```
// fn main() {
//     if x != 4 ||$0 y < 3.14 {}
// }
// ```
// ->
// ```
// fn main() {
//     if !(x == 4 && y >= 3.14) {}
// }
// ```
pub(crate) fn apply_demorgan(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    let mut bin_expr = ctx.find_node_at_offset::<ast::BinExpr>()?;
    let op = bin_expr.op_kind()?;
    let op_range = bin_expr.op_token()?.text_range();

    // Is the cursor on the expression's logical operator?
    if !op_range.contains_range(ctx.selection_trimmed()) {
        return None;
    }

    // Walk up the tree while we have the same binary operator
    while let Some(parent_expr) = bin_expr.syntax().parent().and_then(ast::BinExpr::cast) {
        match parent_expr.op_kind() {
            Some(parent_op) if parent_op == op => {
                bin_expr = parent_expr;
            }
            _ => break,
        }
    }

    let op = bin_expr.op_kind()?;
    let inv_token = match op {
        ast::BinaryOp::LogicOp(ast::LogicOp::And) => SyntaxKind::PIPE2,
        ast::BinaryOp::LogicOp(ast::LogicOp::Or) => SyntaxKind::AMP2,
        _ => return None,
    };

    let demorganed = bin_expr.clone_subtree().clone_for_update();

    ted::replace(demorganed.op_token()?, ast::make::token(inv_token));
    let mut exprs = VecDeque::from([
        (bin_expr.lhs()?, demorganed.lhs()?),
        (bin_expr.rhs()?, demorganed.rhs()?),
    ]);

    while let Some((expr, dm)) = exprs.pop_front() {
        if let BinExpr(bin_expr) = &expr {
            if let BinExpr(cbin_expr) = &dm {
                if op == bin_expr.op_kind()? {
                    ted::replace(cbin_expr.op_token()?, ast::make::token(inv_token));
                    exprs.push_back((bin_expr.lhs()?, cbin_expr.lhs()?));
                    exprs.push_back((bin_expr.rhs()?, cbin_expr.rhs()?));
                } else {
                    let mut inv = invert_boolean_expression(expr);
                    if inv.needs_parens_in(dm.syntax().parent()?) {
                        inv = ast::make::expr_paren(inv).clone_for_update();
                    }
                    ted::replace(dm.syntax(), inv.syntax());
                }
            } else {
                return None;
            }
        } else {
            let mut inv = invert_boolean_expression(dm.clone_subtree()).clone_for_update();
            if inv.needs_parens_in(dm.syntax().parent()?) {
                inv = ast::make::expr_paren(inv).clone_for_update();
            }
            ted::replace(dm.syntax(), inv.syntax());
        }
    }

    acc.add_group(
        &GroupLabel("Apply De Morgan's law".to_owned()),
        AssistId("apply_demorgan", AssistKind::RefactorRewrite),
        "Apply De Morgan's law",
        op_range,
        |edit| {
            let demorganed = ast::Expr::BinExpr(demorganed);
            let paren_expr = bin_expr.syntax().parent().and_then(ast::ParenExpr::cast);
            let neg_expr = paren_expr
                .clone()
                .and_then(|paren_expr| paren_expr.syntax().parent())
                .and_then(ast::PrefixExpr::cast)
                .filter(|prefix_expr| matches!(prefix_expr.op_kind(), Some(ast::UnaryOp::Not)))
                .map(ast::Expr::PrefixExpr);

            if let Some(paren_expr) = paren_expr {
                if let Some(neg_expr) = neg_expr {
                    cov_mark::hit!(demorgan_double_negation);
                    let parent = neg_expr.syntax().parent();

                    if parent.is_some_and(|parent| demorganed.needs_parens_in(parent)) {
                        cov_mark::hit!(demorgan_keep_parens_for_op_precedence2);
                        edit.replace_ast(neg_expr, make::expr_paren(demorganed));
                    } else {
                        edit.replace_ast(neg_expr, demorganed);
                    };
                } else {
                    cov_mark::hit!(demorgan_double_parens);
                    edit.replace_ast(paren_expr.into(), add_bang_paren(demorganed));
                }
            } else {
                edit.replace_ast(bin_expr.into(), add_bang_paren(demorganed));
            }
        },
    )
}

// Assist: apply_demorgan_iterator
//
// Apply https://en.wikipedia.org/wiki/De_Morgan%27s_laws[De Morgan's law] to
// `Iterator::all` and `Iterator::any`.
//
// This transforms expressions of the form `!iter.any(|x| predicate(x))` into
// `iter.all(|x| !predicate(x))` and vice versa. This also works the other way for
// `Iterator::all` into `Iterator::any`.
//
// ```
// # //- minicore: iterator
// fn main() {
//     let arr = [1, 2, 3];
//     if !arr.into_iter().$0any(|num| num == 4) {
//         println!("foo");
//     }
// }
// ```
// ->
// ```
// fn main() {
//     let arr = [1, 2, 3];
//     if arr.into_iter().all(|num| num != 4) {
//         println!("foo");
//     }
// }
// ```
pub(crate) fn apply_demorgan_iterator(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    let method_call: ast::MethodCallExpr = ctx.find_node_at_offset()?;
    let (name, arg_expr) = validate_method_call_expr(ctx, &method_call)?;

    let ast::Expr::ClosureExpr(closure_expr) = arg_expr else { return None };
    let closure_body = closure_expr.body()?;

    let op_range = method_call.syntax().text_range();
    let label = format!("Apply De Morgan's law to `Iterator::{}`", name.text().as_str());
    acc.add_group(
        &GroupLabel("Apply De Morgan's law".to_owned()),
        AssistId("apply_demorgan_iterator", AssistKind::RefactorRewrite),
        label,
        op_range,
        |edit| {
            // replace the method name
            let new_name = match name.text().as_str() {
                "all" => make::name_ref("any"),
                "any" => make::name_ref("all"),
                _ => unreachable!(),
            }
            .clone_for_update();
            edit.replace_ast(name, new_name);

            // negate all tail expressions in the closure body
            let tail_cb = &mut |e: &_| tail_cb_impl(edit, e);
            walk_expr(&closure_body, &mut |expr| {
                if let ast::Expr::ReturnExpr(ret_expr) = expr {
                    if let Some(ret_expr_arg) = &ret_expr.expr() {
                        for_each_tail_expr(ret_expr_arg, tail_cb);
                    }
                }
            });
            for_each_tail_expr(&closure_body, tail_cb);

            // negate the whole method call
            if let Some(prefix_expr) = method_call
                .syntax()
                .parent()
                .and_then(ast::PrefixExpr::cast)
                .filter(|prefix_expr| matches!(prefix_expr.op_kind(), Some(ast::UnaryOp::Not)))
            {
                edit.delete(
                    prefix_expr
                        .op_token()
                        .expect("prefix expression always has an operator")
                        .text_range(),
                );
            } else {
                edit.insert(method_call.syntax().text_range().start(), "!");
            }
        },
    )
}

/// Ensures that the method call is to `Iterator::all` or `Iterator::any`.
fn validate_method_call_expr(
    ctx: &AssistContext<'_>,
    method_call: &ast::MethodCallExpr,
) -> Option<(ast::NameRef, ast::Expr)> {
    let name_ref = method_call.name_ref()?;
    if name_ref.text() != "all" && name_ref.text() != "any" {
        return None;
    }
    let arg_expr = method_call.arg_list()?.args().next()?;

    let sema = &ctx.sema;

    let receiver = method_call.receiver()?;
    let it_type = sema.type_of_expr(&receiver)?.adjusted();
    let module = sema.scope(receiver.syntax())?.module();
    let krate = module.krate();

    let iter_trait = FamousDefs(sema, krate).core_iter_Iterator()?;
    it_type.impls_trait(sema.db, iter_trait, &[]).then_some((name_ref, arg_expr))
}

fn tail_cb_impl(edit: &mut SourceChangeBuilder, e: &ast::Expr) {
    match e {
        ast::Expr::BreakExpr(break_expr) => {
            if let Some(break_expr_arg) = break_expr.expr() {
                for_each_tail_expr(&break_expr_arg, &mut |e| tail_cb_impl(edit, e))
            }
        }
        ast::Expr::ReturnExpr(_) => {
            // all return expressions have already been handled by the walk loop
        }
        e => {
            let inverted_body = invert_boolean_expression(e.clone());
            edit.replace(e.syntax().text_range(), inverted_body.syntax().text());
        }
    }
}

/// Add bang and parentheses to the expression.
fn add_bang_paren(expr: ast::Expr) -> ast::Expr {
    make::expr_prefix(T![!], make::expr_paren(expr))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::{check_assist, check_assist_not_applicable};

    #[test]
    fn demorgan_handles_leq() {
        check_assist(
            apply_demorgan,
            r#"
struct S;
fn f() { S < S &&$0 S <= S }
"#,
            r#"
struct S;
fn f() { !(S >= S || S > S) }
"#,
        );
    }

    #[test]
    fn demorgan_handles_geq() {
        check_assist(
            apply_demorgan,
            r#"
struct S;
fn f() { S > S &&$0 S >= S }
"#,
            r#"
struct S;
fn f() { !(S <= S || S < S) }
"#,
        );
    }

    #[test]
    fn demorgan_turns_and_into_or() {
        check_assist(apply_demorgan, "fn f() { !x &&$0 !x }", "fn f() { !(x || x) }")
    }

    #[test]
    fn demorgan_turns_or_into_and() {
        check_assist(apply_demorgan, "fn f() { !x ||$0 !x }", "fn f() { !(x && x) }")
    }

    #[test]
    fn demorgan_removes_inequality() {
        check_assist(apply_demorgan, "fn f() { x != x ||$0 !x }", "fn f() { !(x == x && x) }")
    }

    #[test]
    fn demorgan_general_case() {
        check_assist(apply_demorgan, "fn f() { x ||$0 x }", "fn f() { !(!x && !x) }")
    }

    #[test]
    fn demorgan_multiple_terms() {
        check_assist(apply_demorgan, "fn f() { x ||$0 y || z }", "fn f() { !(!x && !y && !z) }");
        check_assist(apply_demorgan, "fn f() { x || y ||$0 z }", "fn f() { !(!x && !y && !z) }");
    }

    #[test]
    fn demorgan_doesnt_apply_with_cursor_not_on_op() {
        check_assist_not_applicable(apply_demorgan, "fn f() { $0 !x || !x }")
    }

    #[test]
    fn demorgan_doesnt_double_negation() {
        cov_mark::check!(demorgan_double_negation);
        check_assist(apply_demorgan, "fn f() { !(x ||$0 x) }", "fn f() { !x && !x }")
    }

    #[test]
    fn demorgan_doesnt_double_parens() {
        cov_mark::check!(demorgan_double_parens);
        check_assist(apply_demorgan, "fn f() { (x ||$0 x) }", "fn f() { !(!x && !x) }")
    }

    #[test]
    fn demorgan_doesnt_hang() {
        check_assist(
            apply_demorgan,
            "fn f() { 1 || 3 &&$0 4 || 5 }",
            "fn f() { 1 || !(!3 || !4) || 5 }",
        )
    }

    #[test]
    fn demorgan_keep_pars_for_op_precedence() {
        check_assist(
            apply_demorgan,
            "fn main() {
    let _ = !(!a ||$0 !(b || c));
}
",
            "fn main() {
    let _ = a && (b || c);
}
",
        );
    }

    #[test]
    fn demorgan_keep_pars_for_op_precedence2() {
        cov_mark::check!(demorgan_keep_parens_for_op_precedence2);
        check_assist(
            apply_demorgan,
            "fn f() { (a && !(b &&$0 c); }",
            "fn f() { (a && (!b || !c); }",
        );
    }

    #[test]
    fn demorgan_keep_pars_for_op_precedence3() {
        check_assist(apply_demorgan, "fn f() { (a || !(b &&$0 c); }", "fn f() { (a || !b || !c; }");
    }

    #[test]
    fn demorgan_removes_pars_in_eq_precedence() {
        check_assist(
            apply_demorgan,
            "fn() { let x = a && !(!b |$0| !c); }",
            "fn() { let x = a && b && c; }",
        )
    }

    #[test]
    fn demorgan_removes_pars_for_op_precedence2() {
        check_assist(apply_demorgan, "fn f() { (a || !(b ||$0 c); }", "fn f() { (a || !b && !c; }");
    }

    #[test]
    fn demorgan_iterator_any_all_reverse() {
        check_assist(
            apply_demorgan_iterator,
            r#"
//- minicore: iterator
fn main() {
    let arr = [1, 2, 3];
    if arr.into_iter().all(|num| num $0!= 4) {
        println!("foo");
    }
}
"#,
            r#"
fn main() {
    let arr = [1, 2, 3];
    if !arr.into_iter().any(|num| num == 4) {
        println!("foo");
    }
}
"#,
        );
    }

    #[test]
    fn demorgan_iterator_all_any() {
        check_assist(
            apply_demorgan_iterator,
            r#"
//- minicore: iterator
fn main() {
    let arr = [1, 2, 3];
    if !arr.into_iter().$0all(|num| num > 3) {
        println!("foo");
    }
}
"#,
            r#"
fn main() {
    let arr = [1, 2, 3];
    if arr.into_iter().any(|num| num <= 3) {
        println!("foo");
    }
}
"#,
        );
    }

    #[test]
    fn demorgan_iterator_multiple_terms() {
        check_assist(
            apply_demorgan_iterator,
            r#"
//- minicore: iterator
fn main() {
    let arr = [1, 2, 3];
    if !arr.into_iter().$0any(|num| num > 3 && num == 23 && num <= 30) {
        println!("foo");
    }
}
"#,
            r#"
fn main() {
    let arr = [1, 2, 3];
    if arr.into_iter().all(|num| !(num > 3 && num == 23 && num <= 30)) {
        println!("foo");
    }
}
"#,
        );
    }

    #[test]
    fn demorgan_iterator_double_negation() {
        check_assist(
            apply_demorgan_iterator,
            r#"
//- minicore: iterator
fn main() {
    let arr = [1, 2, 3];
    if !arr.into_iter().$0all(|num| !(num > 3)) {
        println!("foo");
    }
}
"#,
            r#"
fn main() {
    let arr = [1, 2, 3];
    if arr.into_iter().any(|num| num > 3) {
        println!("foo");
    }
}
"#,
        );
    }

    #[test]
    fn demorgan_iterator_double_parens() {
        check_assist(
            apply_demorgan_iterator,
            r#"
//- minicore: iterator
fn main() {
    let arr = [1, 2, 3];
    if !arr.into_iter().$0any(|num| (num > 3 && (num == 1 || num == 2))) {
        println!("foo");
    }
}
"#,
            r#"
fn main() {
    let arr = [1, 2, 3];
    if arr.into_iter().all(|num| !(num > 3 && (num == 1 || num == 2))) {
        println!("foo");
    }
}
"#,
        );
    }

    #[test]
    fn demorgan_iterator_multiline() {
        check_assist(
            apply_demorgan_iterator,
            r#"
//- minicore: iterator
fn main() {
    let arr = [1, 2, 3];
    if arr
        .into_iter()
        .all$0(|num| !num.is_negative())
    {
        println!("foo");
    }
}
"#,
            r#"
fn main() {
    let arr = [1, 2, 3];
    if !arr
        .into_iter()
        .any(|num| num.is_negative())
    {
        println!("foo");
    }
}
"#,
        );
    }

    #[test]
    fn demorgan_iterator_block_closure() {
        check_assist(
            apply_demorgan_iterator,
            r#"
//- minicore: iterator
fn main() {
    let arr = [-1, 1, 2, 3];
    if arr.into_iter().all(|num: i32| {
        $0if num.is_positive() {
            num <= 3
        } else {
            num >= -1
        }
    }) {
        println!("foo");
    }
}
"#,
            r#"
fn main() {
    let arr = [-1, 1, 2, 3];
    if !arr.into_iter().any(|num: i32| {
        if num.is_positive() {
            num > 3
        } else {
            num < -1
        }
    }) {
        println!("foo");
    }
}
"#,
        );
    }

    #[test]
    fn demorgan_iterator_wrong_method() {
        check_assist_not_applicable(
            apply_demorgan_iterator,
            r#"
//- minicore: iterator
fn main() {
    let arr = [1, 2, 3];
    if !arr.into_iter().$0map(|num| num > 3) {
        println!("foo");
    }
}
"#,
        );
    }
}
