use hir::AsName;
use syntax::{
    ast::{self, edit::AstNodeEdit, make},
    AstNode,
};
use test_utils::mark;

use crate::{
    assist_context::{AssistContext, Assists},
    AssistId, AssistKind,
};

// Assist: extract_assignment
//
// Extracts variable assigment to outside an if or match statement.
//
// ```
// fn main() {
//     let mut foo = 6;
//
//     if true {
//         <|>foo = 5;
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
pub(crate) fn extract_assigment(acc: &mut Assists, ctx: &AssistContext) -> Option<()> {
    let name = ctx.find_node_at_offset::<ast::NameRef>()?.as_name();

    let (old_stmt, new_stmt) = if let Some(if_expr) = ctx.find_node_at_offset::<ast::IfExpr>() {
        (
            ast::Expr::cast(if_expr.syntax().to_owned())?,
            exprify_if(&if_expr, &name)?.indent(if_expr.indent_level()),
        )
    } else if let Some(match_expr) = ctx.find_node_at_offset::<ast::MatchExpr>() {
        (ast::Expr::cast(match_expr.syntax().to_owned())?, exprify_match(&match_expr, &name)?)
    } else {
        return None;
    };

    let expr_stmt = make::expr_stmt(new_stmt);

    acc.add(
        AssistId("extract_assignment", AssistKind::RefactorExtract),
        "Extract assignment",
        old_stmt.syntax().text_range(),
        move |edit| {
            edit.replace(old_stmt.syntax().text_range(), format!("{} = {};", name, expr_stmt));
        },
    )
}

fn exprify_match(match_expr: &ast::MatchExpr, name: &hir::Name) -> Option<ast::Expr> {
    let new_arm_list = match_expr
        .match_arm_list()?
        .arms()
        .map(|arm| {
            if let ast::Expr::BlockExpr(block) = arm.expr()? {
                let new_block = exprify_block(&block, name)?.indent(block.indent_level());
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

fn exprify_if(statement: &ast::IfExpr, name: &hir::Name) -> Option<ast::Expr> {
    let then_branch = exprify_block(&statement.then_branch()?, name)?;
    let else_branch = match statement.else_branch()? {
        ast::ElseBranch::Block(ref block) => ast::ElseBranch::Block(exprify_block(block, name)?),
        ast::ElseBranch::IfExpr(expr) => {
            mark::hit!(test_extract_assigment_chained_if);
            ast::ElseBranch::IfExpr(ast::IfExpr::cast(
                exprify_if(&expr, name)?.syntax().to_owned(),
            )?)
        }
    };
    Some(make::expr_if(statement.condition()?, then_branch, Some(else_branch)))
}

fn exprify_block(block: &ast::BlockExpr, name: &hir::Name) -> Option<ast::BlockExpr> {
    if block.expr().is_some() {
        return None;
    }

    let mut stmts: Vec<_> = block.statements().collect();
    let stmt = stmts.pop()?;

    if let ast::Stmt::ExprStmt(stmt) = stmt {
        if let ast::Expr::BinExpr(expr) = stmt.expr()? {
            if expr.op_kind()? == ast::BinOp::Assignment
                && &expr.lhs()?.name_ref()?.as_name() == name
            {
                // The last statement in the block is an assignment to the name we want
                return Some(make::block_expr(stmts, Some(expr.rhs()?)));
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::tests::{check_assist, check_assist_not_applicable};

    #[test]
    fn test_extract_assignment_if() {
        check_assist(
            extract_assigment,
            r#"
fn foo() {
    let mut a = 1;

    if true {
        <|>a = 2;
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
    fn test_extract_assignment_match() {
        check_assist(
            extract_assigment,
            r#"
fn foo() {
    let mut a = 1;

    match 1 {
        1 => {
            <|>a = 2;
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
    fn test_extract_assignment_not_last_not_applicable() {
        check_assist_not_applicable(
            extract_assigment,
            r#"
fn foo() {
    let mut a = 1;

    if true {
        <|>a = 2;
        b = a;
    } else {
        a = 3;
    }
}"#,
        )
    }

    #[test]
    fn test_extract_assignment_chained_if() {
        mark::check!(test_extract_assigment_chained_if);
        check_assist(
            extract_assigment,
            r#"
fn foo() {
    let mut a = 1;

    if true {
        <|>a = 2;
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
    fn test_extract_assigment_retains_stmts() {
        check_assist(
            extract_assigment,
            r#"
fn foo() {
    let mut a = 1;

    if true {
        let b = 2;
        <|>a = 2;
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
    fn extract_assignment_let_stmt_not_applicable() {
        check_assist_not_applicable(
            extract_assigment,
            r#"
fn foo() {
    let mut a = 1;

    let b = if true {
        <|>a = 2
    } else {
        a = 3
    };
}"#,
        )
    }

    #[test]
    fn extract_assignment_if_missing_assigment_not_applicable() {
        check_assist_not_applicable(
            extract_assigment,
            r#"
fn foo() {
    let mut a = 1;

    if true {
        <|>a = 2;
    } else {}
}"#,
        )
    }

    #[test]
    fn extract_assignment_match_missing_assigment_not_applicable() {
        check_assist_not_applicable(
            extract_assigment,
            r#"
fn foo() {
    let mut a = 1;

    match 1 {
        1 => {
            <|>a = 2;
        },
        2 => {
            a = 3;
        },
        3 => {},
    }
}"#,
        )
    }
}
