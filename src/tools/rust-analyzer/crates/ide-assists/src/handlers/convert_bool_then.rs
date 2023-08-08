use hir::{known, AsAssocItem, Semantics};
use ide_db::{
    famous_defs::FamousDefs,
    syntax_helpers::node_ext::{
        block_as_lone_tail, for_each_tail_expr, is_pattern_cond, preorder_expr,
    },
    RootDatabase,
};
use itertools::Itertools;
use syntax::{
    ast::{self, edit::AstNodeEdit, make, HasArgList},
    ted, AstNode, SyntaxNode,
};

use crate::{
    utils::{invert_boolean_expression, unwrap_trivial_block},
    AssistContext, AssistId, AssistKind, Assists,
};

// Assist: convert_if_to_bool_then
//
// Converts an if expression into a corresponding `bool::then` call.
//
// ```
// # //- minicore: option
// fn main() {
//     if$0 cond {
//         Some(val)
//     } else {
//         None
//     }
// }
// ```
// ->
// ```
// fn main() {
//     cond.then(|| val)
// }
// ```
pub(crate) fn convert_if_to_bool_then(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    // FIXME applies to match as well
    let expr = ctx.find_node_at_offset::<ast::IfExpr>()?;
    if !expr.if_token()?.text_range().contains_inclusive(ctx.offset()) {
        return None;
    }

    let cond = expr.condition().filter(|cond| !is_pattern_cond(cond.clone()))?;
    let then = expr.then_branch()?;
    let else_ = match expr.else_branch()? {
        ast::ElseBranch::Block(b) => b,
        ast::ElseBranch::IfExpr(_) => {
            cov_mark::hit!(convert_if_to_bool_then_chain);
            return None;
        }
    };

    let (none_variant, some_variant) = option_variants(&ctx.sema, expr.syntax())?;

    let (invert_cond, closure_body) = match (
        block_is_none_variant(&ctx.sema, &then, none_variant),
        block_is_none_variant(&ctx.sema, &else_, none_variant),
    ) {
        (invert @ true, false) => (invert, ast::Expr::BlockExpr(else_)),
        (invert @ false, true) => (invert, ast::Expr::BlockExpr(then)),
        _ => return None,
    };

    if is_invalid_body(&ctx.sema, some_variant, &closure_body) {
        cov_mark::hit!(convert_if_to_bool_then_pattern_invalid_body);
        return None;
    }

    let target = expr.syntax().text_range();
    acc.add(
        AssistId("convert_if_to_bool_then", AssistKind::RefactorRewrite),
        "Convert `if` expression to `bool::then` call",
        target,
        |builder| {
            let closure_body = closure_body.clone_for_update();
            // Rewrite all `Some(e)` in tail position to `e`
            let mut replacements = Vec::new();
            for_each_tail_expr(&closure_body, &mut |e| {
                let e = match e {
                    ast::Expr::BreakExpr(e) => e.expr(),
                    e @ ast::Expr::CallExpr(_) => Some(e.clone()),
                    _ => None,
                };
                if let Some(ast::Expr::CallExpr(call)) = e {
                    if let Some(arg_list) = call.arg_list() {
                        if let Some(arg) = arg_list.args().next() {
                            replacements.push((call.syntax().clone(), arg.syntax().clone()));
                        }
                    }
                }
            });
            replacements.into_iter().for_each(|(old, new)| ted::replace(old, new));
            let closure_body = match closure_body {
                ast::Expr::BlockExpr(block) => unwrap_trivial_block(block),
                e => e,
            };

            let parenthesize = matches!(
                cond,
                ast::Expr::BinExpr(_)
                    | ast::Expr::BlockExpr(_)
                    | ast::Expr::BoxExpr(_)
                    | ast::Expr::BreakExpr(_)
                    | ast::Expr::CastExpr(_)
                    | ast::Expr::ClosureExpr(_)
                    | ast::Expr::ContinueExpr(_)
                    | ast::Expr::ForExpr(_)
                    | ast::Expr::IfExpr(_)
                    | ast::Expr::LoopExpr(_)
                    | ast::Expr::MacroExpr(_)
                    | ast::Expr::MatchExpr(_)
                    | ast::Expr::PrefixExpr(_)
                    | ast::Expr::RangeExpr(_)
                    | ast::Expr::RefExpr(_)
                    | ast::Expr::ReturnExpr(_)
                    | ast::Expr::WhileExpr(_)
                    | ast::Expr::YieldExpr(_)
            );
            let cond = if invert_cond { invert_boolean_expression(cond) } else { cond };
            let cond = if parenthesize { make::expr_paren(cond) } else { cond };
            let arg_list = make::arg_list(Some(make::expr_closure(None, closure_body)));
            let mcall = make::expr_method_call(cond, make::name_ref("then"), arg_list);
            builder.replace(target, mcall.to_string());
        },
    )
}

// Assist: convert_bool_then_to_if
//
// Converts a `bool::then` method call to an equivalent if expression.
//
// ```
// # //- minicore: bool_impl
// fn main() {
//     (0 == 0).then$0(|| val)
// }
// ```
// ->
// ```
// fn main() {
//     if 0 == 0 {
//         Some(val)
//     } else {
//         None
//     }
// }
// ```
pub(crate) fn convert_bool_then_to_if(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    let name_ref = ctx.find_node_at_offset::<ast::NameRef>()?;
    let mcall = name_ref.syntax().parent().and_then(ast::MethodCallExpr::cast)?;
    let receiver = mcall.receiver()?;
    let closure_body = mcall.arg_list()?.args().exactly_one().ok()?;
    let closure_body = match closure_body {
        ast::Expr::ClosureExpr(expr) => expr.body()?,
        _ => return None,
    };
    // Verify this is `bool::then` that is being called.
    let func = ctx.sema.resolve_method_call(&mcall)?;
    if func.name(ctx.sema.db).display(ctx.db()).to_string() != "then" {
        return None;
    }
    let assoc = func.as_assoc_item(ctx.sema.db)?;
    match assoc.container(ctx.sema.db) {
        hir::AssocItemContainer::Impl(impl_) if impl_.self_ty(ctx.sema.db).is_bool() => {}
        _ => return None,
    }

    let target = mcall.syntax().text_range();
    acc.add(
        AssistId("convert_bool_then_to_if", AssistKind::RefactorRewrite),
        "Convert `bool::then` call to `if`",
        target,
        |builder| {
            let closure_body = match closure_body {
                ast::Expr::BlockExpr(block) => block,
                e => make::block_expr(None, Some(e)),
            };

            let closure_body = closure_body.clone_for_update();
            // Wrap all tails in `Some(...)`
            let none_path = make::expr_path(make::ext::ident_path("None"));
            let some_path = make::expr_path(make::ext::ident_path("Some"));
            let mut replacements = Vec::new();
            for_each_tail_expr(&ast::Expr::BlockExpr(closure_body.clone()), &mut |e| {
                let e = match e {
                    ast::Expr::BreakExpr(e) => e.expr(),
                    ast::Expr::ReturnExpr(e) => e.expr(),
                    _ => Some(e.clone()),
                };
                if let Some(expr) = e {
                    replacements.push((
                        expr.syntax().clone(),
                        make::expr_call(some_path.clone(), make::arg_list(Some(expr)))
                            .syntax()
                            .clone_for_update(),
                    ));
                }
            });
            replacements.into_iter().for_each(|(old, new)| ted::replace(old, new));

            let cond = match &receiver {
                ast::Expr::ParenExpr(expr) => expr.expr().unwrap_or(receiver),
                _ => receiver,
            };
            let if_expr = make::expr_if(
                cond,
                closure_body.reset_indent(),
                Some(ast::ElseBranch::Block(make::block_expr(None, Some(none_path)))),
            )
            .indent(mcall.indent_level());

            builder.replace(target, if_expr.to_string());
        },
    )
}

fn option_variants(
    sema: &Semantics<'_, RootDatabase>,
    expr: &SyntaxNode,
) -> Option<(hir::Variant, hir::Variant)> {
    let fam = FamousDefs(sema, sema.scope(expr)?.krate());
    let option_variants = fam.core_option_Option()?.variants(sema.db);
    match &*option_variants {
        &[variant0, variant1] => Some(if variant0.name(sema.db) == known::None {
            (variant0, variant1)
        } else {
            (variant1, variant0)
        }),
        _ => None,
    }
}

/// Traverses the expression checking if it contains `return` or `?` expressions or if any tail is not a `Some(expr)` expression.
/// If any of these conditions are met it is impossible to rewrite this as a `bool::then` call.
fn is_invalid_body(
    sema: &Semantics<'_, RootDatabase>,
    some_variant: hir::Variant,
    expr: &ast::Expr,
) -> bool {
    let mut invalid = false;
    preorder_expr(expr, &mut |e| {
        invalid |=
            matches!(e, syntax::WalkEvent::Enter(ast::Expr::TryExpr(_) | ast::Expr::ReturnExpr(_)));
        invalid
    });
    if !invalid {
        for_each_tail_expr(expr, &mut |e| {
            if invalid {
                return;
            }
            let e = match e {
                ast::Expr::BreakExpr(e) => e.expr(),
                e @ ast::Expr::CallExpr(_) => Some(e.clone()),
                _ => None,
            };
            if let Some(ast::Expr::CallExpr(call)) = e {
                if let Some(ast::Expr::PathExpr(p)) = call.expr() {
                    let res = p.path().and_then(|p| sema.resolve_path(&p));
                    if let Some(hir::PathResolution::Def(hir::ModuleDef::Variant(v))) = res {
                        return invalid |= v != some_variant;
                    }
                }
            }
            invalid = true
        });
    }
    invalid
}

fn block_is_none_variant(
    sema: &Semantics<'_, RootDatabase>,
    block: &ast::BlockExpr,
    none_variant: hir::Variant,
) -> bool {
    block_as_lone_tail(block).and_then(|e| match e {
        ast::Expr::PathExpr(pat) => match sema.resolve_path(&pat.path()?)? {
            hir::PathResolution::Def(hir::ModuleDef::Variant(v)) => Some(v),
            _ => None,
        },
        _ => None,
    }) == Some(none_variant)
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_assist, check_assist_not_applicable};

    use super::*;

    #[test]
    fn convert_if_to_bool_then_simple() {
        check_assist(
            convert_if_to_bool_then,
            r"
//- minicore:option
fn main() {
    if$0 true {
        Some(15)
    } else {
        None
    }
}
",
            r"
fn main() {
    true.then(|| 15)
}
",
        );
    }

    #[test]
    fn convert_if_to_bool_then_invert() {
        check_assist(
            convert_if_to_bool_then,
            r"
//- minicore:option
fn main() {
    if$0 true {
        None
    } else {
        Some(15)
    }
}
",
            r"
fn main() {
    false.then(|| 15)
}
",
        );
    }

    #[test]
    fn convert_if_to_bool_then_none_none() {
        check_assist_not_applicable(
            convert_if_to_bool_then,
            r"
//- minicore:option
fn main() {
    if$0 true {
        None
    } else {
        None
    }
}
",
        );
    }

    #[test]
    fn convert_if_to_bool_then_some_some() {
        check_assist_not_applicable(
            convert_if_to_bool_then,
            r"
//- minicore:option
fn main() {
    if$0 true {
        Some(15)
    } else {
        Some(15)
    }
}
",
        );
    }

    #[test]
    fn convert_if_to_bool_then_mixed() {
        check_assist_not_applicable(
            convert_if_to_bool_then,
            r"
//- minicore:option
fn main() {
    if$0 true {
        if true {
            Some(15)
        } else {
            None
        }
    } else {
        None
    }
}
",
        );
    }

    #[test]
    fn convert_if_to_bool_then_chain() {
        cov_mark::check!(convert_if_to_bool_then_chain);
        check_assist_not_applicable(
            convert_if_to_bool_then,
            r"
//- minicore:option
fn main() {
    if$0 true {
        Some(15)
    } else if true {
        None
    } else {
        None
    }
}
",
        );
    }

    #[test]
    fn convert_if_to_bool_then_pattern_cond() {
        check_assist_not_applicable(
            convert_if_to_bool_then,
            r"
//- minicore:option
fn main() {
    if$0 let true = true {
        Some(15)
    } else {
        None
    }
}
",
        );
    }

    #[test]
    fn convert_if_to_bool_then_pattern_invalid_body() {
        cov_mark::check_count!(convert_if_to_bool_then_pattern_invalid_body, 2);
        check_assist_not_applicable(
            convert_if_to_bool_then,
            r"
//- minicore:option
fn make_me_an_option() -> Option<i32> { None }
fn main() {
    if$0 true {
        if true {
            make_me_an_option()
        } else {
            Some(15)
        }
    } else {
        None
    }
}
",
        );
        check_assist_not_applicable(
            convert_if_to_bool_then,
            r"
//- minicore:option
fn main() {
    if$0 true {
        if true {
            return;
        }
        Some(15)
    } else {
        None
    }
}
",
        );
    }

    #[test]
    fn convert_bool_then_to_if_inapplicable() {
        check_assist_not_applicable(
            convert_bool_then_to_if,
            r"
//- minicore:bool_impl
fn main() {
    0.t$0hen(|| 15);
}
",
        );
        check_assist_not_applicable(
            convert_bool_then_to_if,
            r"
//- minicore:bool_impl
fn main() {
    true.t$0hen(15);
}
",
        );
        check_assist_not_applicable(
            convert_bool_then_to_if,
            r"
//- minicore:bool_impl
fn main() {
    true.t$0hen(|| 15, 15);
}
",
        );
    }

    #[test]
    fn convert_bool_then_to_if_simple() {
        check_assist(
            convert_bool_then_to_if,
            r"
//- minicore:bool_impl
fn main() {
    true.t$0hen(|| 15)
}
",
            r"
fn main() {
    if true {
        Some(15)
    } else {
        None
    }
}
",
        );
        check_assist(
            convert_bool_then_to_if,
            r"
//- minicore:bool_impl
fn main() {
    true.t$0hen(|| {
        15
    })
}
",
            r"
fn main() {
    if true {
        Some(15)
    } else {
        None
    }
}
",
        );
    }

    #[test]
    fn convert_bool_then_to_if_tails() {
        check_assist(
            convert_bool_then_to_if,
            r"
//- minicore:bool_impl
fn main() {
    true.t$0hen(|| {
        loop {
            if false {
                break 0;
            }
            break 15;
        }
    })
}
",
            r"
fn main() {
    if true {
        loop {
            if false {
                break Some(0);
            }
            break Some(15);
        }
    } else {
        None
    }
}
",
        );
    }
}
