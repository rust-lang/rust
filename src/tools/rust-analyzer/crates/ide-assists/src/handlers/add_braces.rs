use either::Either;
use syntax::{
    AstNode, T,
    ast::{self, edit::AstNodeEdit, syntax_factory::SyntaxFactory},
    match_ast,
};

use crate::{AssistContext, AssistId, Assists};

// Assist: add_braces
//
// Adds braces to closure bodies, match arm expressions and assignment bodies.
//
// ```
// fn foo(n: i32) -> i32 {
//     match n {
//         1 =>$0 n + 1,
//         _ => 0
//     }
// }
// ```
// ->
// ```
// fn foo(n: i32) -> i32 {
//     match n {
//         1 => {
//             n + 1
//         },
//         _ => 0
//     }
// }
// ```
// ---
// ```
// fn foo(n: i32) -> i32 {
//     let x =$0 n + 2;
// }
// ```
// ->
// ```
// fn foo(n: i32) -> i32 {
//     let x = {
//         n + 2
//     };
// }
// ```
pub(crate) fn add_braces(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    let (expr_type, expr) = get_replacement_node(ctx)?;

    acc.add(
        AssistId::refactor_rewrite("add_braces"),
        match expr_type {
            ParentType::ClosureExpr => "Add braces to this closure body",
            ParentType::MatchArmExpr => "Add braces to this match arm expression",
            ParentType::Assignment => "Add braces to this assignment expression",
        },
        expr.syntax().text_range(),
        |builder| {
            let make = SyntaxFactory::with_mappings();
            let mut editor = builder.make_editor(expr.syntax());

            let new_expr = expr.reset_indent().indent(1.into());
            let block_expr = make.block_expr(None, Some(new_expr));

            editor.replace(expr.syntax(), block_expr.indent(expr.indent_level()).syntax());

            editor.add_mappings(make.finish_with_mappings());
            builder.add_file_edits(ctx.vfs_file_id(), editor);
        },
    )
}

enum ParentType {
    MatchArmExpr,
    ClosureExpr,
    Assignment,
}

fn get_replacement_node(ctx: &AssistContext<'_>) -> Option<(ParentType, ast::Expr)> {
    let node = ctx.find_node_at_offset::<Either<ast::MatchArm, ast::ClosureExpr>>();
    let (parent_type, body) = if let Some(eq_token) = ctx.find_token_syntax_at_offset(T![=]) {
        let parent = eq_token.parent()?;
        let body = match_ast! {
            match parent {
                ast::LetStmt(it) => it.initializer()?,
                ast::LetExpr(it) => it.expr()?,
                ast::Static(it) => it.body()?,
                ast::Const(it) => it.body()?,
                _ => return None,
            }
        };
        (ParentType::Assignment, body)
    } else if let Some(Either::Left(match_arm)) = &node {
        let match_arm_expr = match_arm.expr()?;
        (ParentType::MatchArmExpr, match_arm_expr)
    } else if let Some(Either::Right(closure_expr)) = &node {
        let body = closure_expr.body()?;
        (ParentType::ClosureExpr, body)
    } else {
        return None;
    };

    if matches!(body, ast::Expr::BlockExpr(_)) {
        return None;
    }

    Some((parent_type, body))
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_assist, check_assist_not_applicable};

    use super::*;

    #[test]
    fn suggest_add_braces_for_closure() {
        check_assist(
            add_braces,
            r#"
fn foo() {
    t(|n|$0 n + 100);
}
"#,
            r#"
fn foo() {
    t(|n| {
        n + 100
    });
}
"#,
        );
    }

    #[test]
    fn suggest_add_braces_for_closure_in_match() {
        check_assist(
            add_braces,
            r#"
fn foo() {
    match () {
        () => {
            t(|n|$0 n + 100);
        }
    }
}
"#,
            r#"
fn foo() {
    match () {
        () => {
            t(|n| {
                n + 100
            });
        }
    }
}
"#,
        );
    }

    #[test]
    fn suggest_add_braces_for_assignment() {
        check_assist(
            add_braces,
            r#"
fn foo() {
    let x =$0 n + 100;
}
"#,
            r#"
fn foo() {
    let x = {
        n + 100
    };
}
"#,
        );
    }

    #[test]
    fn no_assist_for_closures_with_braces() {
        check_assist_not_applicable(
            add_braces,
            r#"
fn foo() {
    t(|n|$0 { n + 100 });
}
"#,
        );
    }

    #[test]
    fn suggest_add_braces_for_match() {
        check_assist(
            add_braces,
            r#"
fn foo() {
    match n {
        Some(n) $0=> 29,
        _ => ()
    };
}
"#,
            r#"
fn foo() {
    match n {
        Some(n) => {
            29
        },
        _ => ()
    };
}
"#,
        );
    }

    #[test]
    fn multiple_indent() {
        check_assist(
            add_braces,
            r#"
fn foo() {
    {
        match n {
            Some(n) $0=> foo(
                29,
                30,
            ),
            _ => ()
        };
    }
}
"#,
            r#"
fn foo() {
    {
        match n {
            Some(n) => {
                foo(
                    29,
                    30,
                )
            },
            _ => ()
        };
    }
}
"#,
        );
    }

    #[test]
    fn no_assist_for_match_with_braces() {
        check_assist_not_applicable(
            add_braces,
            r#"
fn foo() {
    match n {
        Some(n) $0=> { return 29; },
        _ => ()
    };
}
"#,
        );
    }
}
