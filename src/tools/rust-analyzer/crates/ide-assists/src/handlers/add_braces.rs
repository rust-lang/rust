use syntax::{
    AstNode,
    ast::{self, edit_in_place::Indent, syntax_factory::SyntaxFactory},
};

use crate::{AssistContext, AssistId, Assists};

// Assist: add_braces
//
// Adds braces to closure bodies and match arm expressions.
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
pub(crate) fn add_braces(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    let (expr_type, expr) = get_replacement_node(ctx)?;

    acc.add(
        AssistId::refactor_rewrite("add_braces"),
        match expr_type {
            ParentType::ClosureExpr => "Add braces to this closure body",
            ParentType::MatchArmExpr => "Add braces to this match arm expression",
        },
        expr.syntax().text_range(),
        |builder| {
            let make = SyntaxFactory::with_mappings();
            let mut editor = builder.make_editor(expr.syntax());

            let block_expr = make.block_expr(None, Some(expr.clone()));
            block_expr.indent(expr.indent_level());

            editor.replace(expr.syntax(), block_expr.syntax());

            editor.add_mappings(make.finish_with_mappings());
            builder.add_file_edits(ctx.vfs_file_id(), editor);
        },
    )
}

enum ParentType {
    MatchArmExpr,
    ClosureExpr,
}

fn get_replacement_node(ctx: &AssistContext<'_>) -> Option<(ParentType, ast::Expr)> {
    if let Some(match_arm) = ctx.find_node_at_offset::<ast::MatchArm>() {
        let match_arm_expr = match_arm.expr()?;

        if matches!(match_arm_expr, ast::Expr::BlockExpr(_)) {
            return None;
        }

        return Some((ParentType::MatchArmExpr, match_arm_expr));
    } else if let Some(closure_expr) = ctx.find_node_at_offset::<ast::ClosureExpr>() {
        let body = closure_expr.body()?;

        if matches!(body, ast::Expr::BlockExpr(_)) {
            return None;
        }

        return Some((ParentType::ClosureExpr, body));
    }

    None
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
