use syntax::{
    ast::{self, make},
    ted, AstNode,
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
pub(crate) fn pull_assignment_up(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    let assign_expr = ctx.find_node_at_offset::<ast::BinExpr>()?;

    let op_kind = assign_expr.op_kind()?;
    if op_kind != (ast::BinaryOp::Assignment { op: None }) {
        cov_mark::hit!(test_cant_pull_non_assignments);
        return None;
    }

    let mut collector = AssignmentsCollector {
        sema: &ctx.sema,
        common_lhs: assign_expr.lhs()?,
        assignments: Vec::new(),
    };

    let tgt: ast::Expr = if let Some(if_expr) = ctx.find_node_at_offset::<ast::IfExpr>() {
        collector.collect_if(&if_expr)?;
        if_expr.into()
    } else if let Some(match_expr) = ctx.find_node_at_offset::<ast::MatchExpr>() {
        collector.collect_match(&match_expr)?;
        match_expr.into()
    } else {
        return None;
    };

    if let Some(parent) = tgt.syntax().parent() {
        if matches!(parent.kind(), syntax::SyntaxKind::BIN_EXPR | syntax::SyntaxKind::LET_STMT) {
            return None;
        }
    }

    acc.add(
        AssistId("pull_assignment_up", AssistKind::RefactorExtract),
        "Pull assignment up",
        tgt.syntax().text_range(),
        move |edit| {
            let assignments: Vec<_> = collector
                .assignments
                .into_iter()
                .map(|(stmt, rhs)| (edit.make_mut(stmt), rhs.clone_for_update()))
                .collect();

            let tgt = edit.make_mut(tgt);

            for (stmt, rhs) in assignments {
                let mut stmt = stmt.syntax().clone();
                if let Some(parent) = stmt.parent() {
                    if ast::ExprStmt::cast(parent.clone()).is_some() {
                        stmt = parent.clone();
                    }
                }
                ted::replace(stmt, rhs.syntax());
            }
            let assign_expr = make::expr_assignment(collector.common_lhs, tgt.clone());
            let assign_stmt = make::expr_stmt(assign_expr);

            ted::replace(tgt.syntax(), assign_stmt.syntax().clone_for_update());
        },
    )
}

struct AssignmentsCollector<'a> {
    sema: &'a hir::Semantics<'a, ide_db::RootDatabase>,
    common_lhs: ast::Expr,
    assignments: Vec<(ast::BinExpr, ast::Expr)>,
}

impl<'a> AssignmentsCollector<'a> {
    fn collect_match(&mut self, match_expr: &ast::MatchExpr) -> Option<()> {
        for arm in match_expr.match_arm_list()?.arms() {
            match arm.expr()? {
                ast::Expr::BlockExpr(block) => self.collect_block(&block)?,
                ast::Expr::BinExpr(expr) => self.collect_expr(&expr)?,
                _ => return None,
            }
        }

        Some(())
    }
    fn collect_if(&mut self, if_expr: &ast::IfExpr) -> Option<()> {
        let then_branch = if_expr.then_branch()?;
        self.collect_block(&then_branch)?;

        match if_expr.else_branch()? {
            ast::ElseBranch::Block(block) => self.collect_block(&block),
            ast::ElseBranch::IfExpr(expr) => {
                cov_mark::hit!(test_pull_assignment_up_chained_if);
                self.collect_if(&expr)
            }
        }
    }
    fn collect_block(&mut self, block: &ast::BlockExpr) -> Option<()> {
        let last_expr = block.tail_expr().or_else(|| match block.statements().last()? {
            ast::Stmt::ExprStmt(stmt) => stmt.expr(),
            ast::Stmt::Item(_) | ast::Stmt::LetStmt(_) => None,
        })?;

        if let ast::Expr::BinExpr(expr) = last_expr {
            return self.collect_expr(&expr);
        }

        None
    }

    fn collect_expr(&mut self, expr: &ast::BinExpr) -> Option<()> {
        if expr.op_kind()? == (ast::BinaryOp::Assignment { op: None })
            && is_equivalent(self.sema, &expr.lhs()?, &self.common_lhs)
        {
            self.assignments.push((expr.clone(), expr.rhs()?));
            return Some(());
        }
        None
    }
}

fn is_equivalent(
    sema: &hir::Semantics<'_, ide_db::RootDatabase>,
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
            if prefix0.op_kind() == Some(ast::UnaryOp::Deref)
                && prefix1.op_kind() == Some(ast::UnaryOp::Deref) =>
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
    fn test_pull_assignment_up_assignment_expressions() {
        check_assist(
            pull_assignment_up,
            r#"
fn foo() {
    let mut a = 1;

    match 1 {
        1 => { $0a = 2; },
        2 => a = 3,
        3 => {
            a = 4
        }
    }
}"#,
            r#"
fn foo() {
    let mut a = 1;

    a = match 1 {
        1 => { 2 },
        2 => 3,
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
    fn pull_assignment_up_if_missing_assignment_not_applicable() {
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
    fn pull_assignment_up_match_missing_assignment_not_applicable() {
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

    #[test]
    fn test_cant_pull_non_assignments() {
        cov_mark::check!(test_cant_pull_non_assignments);
        check_assist_not_applicable(
            pull_assignment_up,
            r#"
fn foo() {
    let mut a = 1;
    let b = &mut a;

    if true {
        $0*b + 2;
    } else {
        *b + 3;
    }
}
"#,
        )
    }
}
