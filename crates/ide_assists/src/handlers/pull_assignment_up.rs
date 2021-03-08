use syntax::{
    ast::{self, edit::AstNodeEdit, make},
    AstNode,
};

use crate::{
    assist_context::{AssistContext, Assists},
    AssistId, AssistKind,
};

// Assist: pull_assignment_up
//
// Extracts variable assignment to outside an if or match statement.
//
// ```
// fn main() {
//     let mut foo = 6;
//
//     if true {
//         $0foo = 5;
//     } else {
//         foo = 4;
//     }
// }
// ```
// ->
// ```
// fn main() {
//     let mut foo = 6;
//
//     foo = if true {
//         5
//     } else {
//         4
//     };
// }
// ```
pub(crate) fn pull_assignment_up(acc: &mut Assists, ctx: &AssistContext) -> Option<()> {
    let assign_expr = ctx.find_node_at_offset::<ast::BinExpr>()?;
    let name_expr = if assign_expr.op_kind()? == ast::BinOp::Assignment {
        assign_expr.lhs()?
    } else {
        return None;
    };

    let (old_stmt, new_stmt) = if let Some(if_expr) = ctx.find_node_at_offset::<ast::IfExpr>() {
        (
            ast::Expr::cast(if_expr.syntax().to_owned())?,
            exprify_if(&if_expr, &ctx.sema, &name_expr)?.indent(if_expr.indent_level()),
        )
    } else if let Some(match_expr) = ctx.find_node_at_offset::<ast::MatchExpr>() {
        (
            ast::Expr::cast(match_expr.syntax().to_owned())?,
            exprify_match(&match_expr, &ctx.sema, &name_expr)?,
        )
    } else {
        return None;
    };

    let expr_stmt = make::expr_stmt(new_stmt);

    acc.add(
        AssistId("pull_assignment_up", AssistKind::RefactorExtract),
        "Pull assignment up",
        old_stmt.syntax().text_range(),
        move |edit| {
            edit.replace(old_stmt.syntax().text_range(), format!("{} = {};", name_expr, expr_stmt));
        },
    )
}

fn exprify_match(
    match_expr: &ast::MatchExpr,
    sema: &hir::Semantics<ide_db::RootDatabase>,
    name: &ast::Expr,
) -> Option<ast::Expr> {
    let new_arm_list = match_expr
        .match_arm_list()?
        .arms()
        .map(|arm| {
            if let ast::Expr::BlockExpr(block) = arm.expr()? {
                let new_block = exprify_block(&block, sema, name)?.indent(block.indent_level());
                Some(arm.replace_descendant(block, new_block))
            } else {
                None
            }
        })
        .collect::<Option<Vec<_>>>()?;
    let new_arm_list = match_expr
        .match_arm_list()?
        .replace_descendants(match_expr.match_arm_list()?.arms().zip(new_arm_list));
    Some(make::expr_match(match_expr.expr()?, new_arm_list))
}

fn exprify_if(
    statement: &ast::IfExpr,
    sema: &hir::Semantics<ide_db::RootDatabase>,
    name: &ast::Expr,
) -> Option<ast::Expr> {
    let then_branch = exprify_block(&statement.then_branch()?, sema, name)?;
    let else_branch = match statement.else_branch()? {
        ast::ElseBranch::Block(ref block) => {
            ast::ElseBranch::Block(exprify_block(block, sema, name)?)
        }
        ast::ElseBranch::IfExpr(expr) => {
            cov_mark::hit!(test_pull_assignment_up_chained_if);
            ast::ElseBranch::IfExpr(ast::IfExpr::cast(
                exprify_if(&expr, sema, name)?.syntax().to_owned(),
            )?)
        }
    };
    Some(make::expr_if(statement.condition()?, then_branch, Some(else_branch)))
}

fn exprify_block(
    block: &ast::BlockExpr,
    sema: &hir::Semantics<ide_db::RootDatabase>,
    name: &ast::Expr,
) -> Option<ast::BlockExpr> {
    if block.tail_expr().is_some() {
        return None;
    }

    let mut stmts: Vec<_> = block.statements().collect();
    let stmt = stmts.pop()?;

    if let ast::Stmt::ExprStmt(stmt) = stmt {
        if let ast::Expr::BinExpr(expr) = stmt.expr()? {
            if expr.op_kind()? == ast::BinOp::Assignment && is_equivalent(sema, &expr.lhs()?, name)
            {
                // The last statement in the block is an assignment to the name we want
                return Some(make::block_expr(stmts, Some(expr.rhs()?)));
            }
        }
    }
    None
}

fn is_equivalent(
    sema: &hir::Semantics<ide_db::RootDatabase>,
    expr0: &ast::Expr,
    expr1: &ast::Expr,
) -> bool {
    match (expr0, expr1) {
        (ast::Expr::FieldExpr(field_expr0), ast::Expr::FieldExpr(field_expr1)) => {
            cov_mark::hit!(test_pull_assignment_up_field_assignment);
            sema.resolve_field(field_expr0) == sema.resolve_field(field_expr1)
        }
        (ast::Expr::PathExpr(path0), ast::Expr::PathExpr(path1)) => {
            let path0 = path0.path();
            let path1 = path1.path();
            if let (Some(path0), Some(path1)) = (path0, path1) {
                sema.resolve_path(&path0) == sema.resolve_path(&path1)
            } else {
                false
            }
        }
        (ast::Expr::PrefixExpr(prefix0), ast::Expr::PrefixExpr(prefix1))
            if prefix0.op_kind() == Some(ast::PrefixOp::Deref)
                && prefix1.op_kind() == Some(ast::PrefixOp::Deref) =>
        {
            cov_mark::hit!(test_pull_assignment_up_deref);
            if let (Some(prefix0), Some(prefix1)) = (prefix0.expr(), prefix1.expr()) {
                is_equivalent(sema, &prefix0, &prefix1)
            } else {
                false
            }
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::tests::{check_assist, check_assist_not_applicable};

    #[test]
    fn test_pull_assignment_up_if() {
        check_assist(
            pull_assignment_up,
            r#"
fn foo() {
    let mut a = 1;

    if true {
        $0a = 2;
    } else {
        a = 3;
    }
}"#,
            r#"
fn foo() {
    let mut a = 1;

    a = if true {
        2
    } else {
        3
    };
}"#,
        );
    }

    #[test]
    fn test_pull_assignment_up_match() {
        check_assist(
            pull_assignment_up,
            r#"
fn foo() {
    let mut a = 1;

    match 1 {
        1 => {
            $0a = 2;
        },
        2 => {
            a = 3;
        },
        3 => {
            a = 4;
        }
    }
}"#,
            r#"
fn foo() {
    let mut a = 1;

    a = match 1 {
        1 => {
            2
        },
        2 => {
            3
        },
        3 => {
            4
        }
    };
}"#,
        );
    }

    #[test]
    fn test_pull_assignment_up_not_last_not_applicable() {
        check_assist_not_applicable(
            pull_assignment_up,
            r#"
fn foo() {
    let mut a = 1;

    if true {
        $0a = 2;
        b = a;
    } else {
        a = 3;
    }
}"#,
        )
    }

    #[test]
    fn test_pull_assignment_up_chained_if() {
        cov_mark::check!(test_pull_assignment_up_chained_if);
        check_assist(
            pull_assignment_up,
            r#"
fn foo() {
    let mut a = 1;

    if true {
        $0a = 2;
    } else if false {
        a = 3;
    } else {
        a = 4;
    }
}"#,
            r#"
fn foo() {
    let mut a = 1;

    a = if true {
        2
    } else if false {
        3
    } else {
        4
    };
}"#,
        );
    }

    #[test]
    fn test_pull_assignment_up_retains_stmts() {
        check_assist(
            pull_assignment_up,
            r#"
fn foo() {
    let mut a = 1;

    if true {
        let b = 2;
        $0a = 2;
    } else {
        let b = 3;
        a = 3;
    }
}"#,
            r#"
fn foo() {
    let mut a = 1;

    a = if true {
        let b = 2;
        2
    } else {
        let b = 3;
        3
    };
}"#,
        )
    }

    #[test]
    fn pull_assignment_up_let_stmt_not_applicable() {
        check_assist_not_applicable(
            pull_assignment_up,
            r#"
fn foo() {
    let mut a = 1;

    let b = if true {
        $0a = 2
    } else {
        a = 3
    };
}"#,
        )
    }

    #[test]
    fn pull_assignment_up_if_missing_assigment_not_applicable() {
        check_assist_not_applicable(
            pull_assignment_up,
            r#"
fn foo() {
    let mut a = 1;

    if true {
        $0a = 2;
    } else {}
}"#,
        )
    }

    #[test]
    fn pull_assignment_up_match_missing_assigment_not_applicable() {
        check_assist_not_applicable(
            pull_assignment_up,
            r#"
fn foo() {
    let mut a = 1;

    match 1 {
        1 => {
            $0a = 2;
        },
        2 => {
            a = 3;
        },
        3 => {},
    }
}"#,
        )
    }

    #[test]
    fn test_pull_assignment_up_field_assignment() {
        cov_mark::check!(test_pull_assignment_up_field_assignment);
        check_assist(
            pull_assignment_up,
            r#"
struct A(usize);

fn foo() {
    let mut a = A(1);

    if true {
        $0a.0 = 2;
    } else {
        a.0 = 3;
    }
}"#,
            r#"
struct A(usize);

fn foo() {
    let mut a = A(1);

    a.0 = if true {
        2
    } else {
        3
    };
}"#,
        )
    }

    #[test]
    fn test_pull_assignment_up_deref() {
        cov_mark::check!(test_pull_assignment_up_deref);
        check_assist(
            pull_assignment_up,
            r#"
fn foo() {
    let mut a = 1;
    let b = &mut a;

    if true {
        $0*b = 2;
    } else {
        *b = 3;
    }
}
"#,
            r#"
fn foo() {
    let mut a = 1;
    let b = &mut a;

    *b = if true {
        2
    } else {
        3
    };
}
"#,
        )
    }
}
