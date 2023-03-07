use ide_db::defs::{Definition, NameRefClass};
use syntax::{
    ast::{self, HasName},
    ted, AstNode, SyntaxNode,
};

use crate::{
    assist_context::{AssistContext, Assists},
    AssistId, AssistKind,
};

// Assist: convert_match_to_let_else
//
// Converts let statement with match initializer to let-else statement.
//
// ```
// # //- minicore: option
// fn foo(opt: Option<()>) {
//     let val = $0match opt {
//         Some(it) => it,
//         None => return,
//     };
// }
// ```
// ->
// ```
// fn foo(opt: Option<()>) {
//     let Some(val) = opt else { return };
// }
// ```
pub(crate) fn convert_match_to_let_else(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    let let_stmt: ast::LetStmt = ctx.find_node_at_offset()?;
    let binding = let_stmt.pat()?;

    let Some(ast::Expr::MatchExpr(initializer)) = let_stmt.initializer() else { return None };
    let initializer_expr = initializer.expr()?;

    let Some((extracting_arm, diverging_arm)) = find_arms(ctx, &initializer) else { return None };
    if extracting_arm.guard().is_some() {
        cov_mark::hit!(extracting_arm_has_guard);
        return None;
    }

    let diverging_arm_expr = match diverging_arm.expr()? {
        ast::Expr::BlockExpr(block) if block.modifier().is_none() && block.label().is_none() => {
            block.to_string()
        }
        other => format!("{{ {other} }}"),
    };
    let extracting_arm_pat = extracting_arm.pat()?;
    let extracted_variable = find_extracted_variable(ctx, &extracting_arm)?;

    acc.add(
        AssistId("convert_match_to_let_else", AssistKind::RefactorRewrite),
        "Convert match to let-else",
        let_stmt.syntax().text_range(),
        |builder| {
            let extracting_arm_pat =
                rename_variable(&extracting_arm_pat, extracted_variable, binding);
            builder.replace(
                let_stmt.syntax().text_range(),
                format!("let {extracting_arm_pat} = {initializer_expr} else {diverging_arm_expr};"),
            )
        },
    )
}

// Given a match expression, find extracting and diverging arms.
fn find_arms(
    ctx: &AssistContext<'_>,
    match_expr: &ast::MatchExpr,
) -> Option<(ast::MatchArm, ast::MatchArm)> {
    let arms = match_expr.match_arm_list()?.arms().collect::<Vec<_>>();
    if arms.len() != 2 {
        return None;
    }

    let mut extracting = None;
    let mut diverging = None;
    for arm in arms {
        if ctx.sema.type_of_expr(&arm.expr()?)?.original().is_never() {
            diverging = Some(arm);
        } else {
            extracting = Some(arm);
        }
    }

    match (extracting, diverging) {
        (Some(extracting), Some(diverging)) => Some((extracting, diverging)),
        _ => {
            cov_mark::hit!(non_diverging_match);
            None
        }
    }
}

// Given an extracting arm, find the extracted variable.
fn find_extracted_variable(ctx: &AssistContext<'_>, arm: &ast::MatchArm) -> Option<ast::Name> {
    match arm.expr()? {
        ast::Expr::PathExpr(path) => {
            let name_ref = path.syntax().descendants().find_map(ast::NameRef::cast)?;
            match NameRefClass::classify(&ctx.sema, &name_ref)? {
                NameRefClass::Definition(Definition::Local(local)) => {
                    let source = local.source(ctx.db()).value.left()?;
                    Some(source.name()?)
                }
                _ => None,
            }
        }
        _ => {
            cov_mark::hit!(extracting_arm_is_not_an_identity_expr);
            return None;
        }
    }
}

// Rename `extracted` with `binding` in `pat`.
fn rename_variable(pat: &ast::Pat, extracted: ast::Name, binding: ast::Pat) -> SyntaxNode {
    let syntax = pat.syntax().clone_for_update();
    let extracted_syntax = syntax.covering_element(extracted.syntax().text_range());

    // If `extracted` variable is a record field, we should rename it to `binding`,
    // otherwise we just need to replace `extracted` with `binding`.

    if let Some(record_pat_field) = extracted_syntax.ancestors().find_map(ast::RecordPatField::cast)
    {
        if let Some(name_ref) = record_pat_field.field_name() {
            ted::replace(
                record_pat_field.syntax(),
                ast::make::record_pat_field(ast::make::name_ref(&name_ref.text()), binding)
                    .syntax()
                    .clone_for_update(),
            );
        }
    } else {
        ted::replace(extracted_syntax, binding.syntax().clone_for_update());
    }

    syntax
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_assist, check_assist_not_applicable};

    use super::*;

    #[test]
    fn should_not_be_applicable_for_non_diverging_match() {
        cov_mark::check!(non_diverging_match);
        check_assist_not_applicable(
            convert_match_to_let_else,
            r#"
//- minicore: option
fn foo(opt: Option<()>) {
    let val = $0match opt {
        Some(it) => it,
        None => (),
    };
}
"#,
        );
    }

    #[test]
    fn should_not_be_applicable_if_extracting_arm_is_not_an_identity_expr() {
        cov_mark::check_count!(extracting_arm_is_not_an_identity_expr, 2);
        check_assist_not_applicable(
            convert_match_to_let_else,
            r#"
//- minicore: option
fn foo(opt: Option<i32>) {
    let val = $0match opt {
        Some(it) => it + 1,
        None => return,
    };
}
"#,
        );

        check_assist_not_applicable(
            convert_match_to_let_else,
            r#"
//- minicore: option
fn foo(opt: Option<()>) {
    let val = $0match opt {
        Some(it) => {
            let _ = 1 + 1;
            it
        },
        None => return,
    };
}
"#,
        );
    }

    #[test]
    fn should_not_be_applicable_if_extracting_arm_has_guard() {
        cov_mark::check!(extracting_arm_has_guard);
        check_assist_not_applicable(
            convert_match_to_let_else,
            r#"
//- minicore: option
fn foo(opt: Option<()>) {
    let val = $0match opt {
        Some(it) if 2 > 1 => it,
        None => return,
    };
}
"#,
        );
    }

    #[test]
    fn basic_pattern() {
        check_assist(
            convert_match_to_let_else,
            r#"
//- minicore: option
fn foo(opt: Option<()>) {
    let val = $0match opt {
        Some(it) => it,
        None => return,
    };
}
    "#,
            r#"
fn foo(opt: Option<()>) {
    let Some(val) = opt else { return };
}
    "#,
        );
    }

    #[test]
    fn keeps_modifiers() {
        check_assist(
            convert_match_to_let_else,
            r#"
//- minicore: option
fn foo(opt: Option<()>) {
    let ref mut val = $0match opt {
        Some(it) => it,
        None => return,
    };
}
    "#,
            r#"
fn foo(opt: Option<()>) {
    let Some(ref mut val) = opt else { return };
}
    "#,
        );
    }

    #[test]
    fn nested_pattern() {
        check_assist(
            convert_match_to_let_else,
            r#"
//- minicore: option, result
fn foo(opt: Option<Result<()>>) {
    let val = $0match opt {
        Some(Ok(it)) => it,
        _ => return,
    };
}
    "#,
            r#"
fn foo(opt: Option<Result<()>>) {
    let Some(Ok(val)) = opt else { return };
}
    "#,
        );
    }

    #[test]
    fn works_with_any_diverging_block() {
        check_assist(
            convert_match_to_let_else,
            r#"
//- minicore: option
fn foo(opt: Option<()>) {
    loop {
        let val = $0match opt {
            Some(it) => it,
            None => break,
        };
    }
}
    "#,
            r#"
fn foo(opt: Option<()>) {
    loop {
        let Some(val) = opt else { break };
    }
}
    "#,
        );

        check_assist(
            convert_match_to_let_else,
            r#"
//- minicore: option
fn foo(opt: Option<()>) {
    loop {
        let val = $0match opt {
            Some(it) => it,
            None => continue,
        };
    }
}
    "#,
            r#"
fn foo(opt: Option<()>) {
    loop {
        let Some(val) = opt else { continue };
    }
}
    "#,
        );

        check_assist(
            convert_match_to_let_else,
            r#"
//- minicore: option
fn panic() -> ! {}

fn foo(opt: Option<()>) {
    loop {
        let val = $0match opt {
            Some(it) => it,
            None => panic(),
        };
    }
}
    "#,
            r#"
fn panic() -> ! {}

fn foo(opt: Option<()>) {
    loop {
        let Some(val) = opt else { panic() };
    }
}
    "#,
        );
    }

    #[test]
    fn struct_pattern() {
        check_assist(
            convert_match_to_let_else,
            r#"
//- minicore: option
struct Point {
    x: i32,
    y: i32,
}

fn foo(opt: Option<Point>) {
    let val = $0match opt {
        Some(Point { x: 0, y }) => y,
        _ => return,
    };
}
    "#,
            r#"
struct Point {
    x: i32,
    y: i32,
}

fn foo(opt: Option<Point>) {
    let Some(Point { x: 0, y: val }) = opt else { return };
}
    "#,
        );
    }

    #[test]
    fn renames_whole_binding() {
        check_assist(
            convert_match_to_let_else,
            r#"
//- minicore: option
fn foo(opt: Option<i32>) -> Option<i32> {
    let val = $0match opt {
        it @ Some(42) => it,
        _ => return None,
    };
    val
}
    "#,
            r#"
fn foo(opt: Option<i32>) -> Option<i32> {
    let val @ Some(42) = opt else { return None };
    val
}
    "#,
        );
    }

    #[test]
    fn complex_pattern() {
        check_assist(
            convert_match_to_let_else,
            r#"
//- minicore: option
fn f() {
    let (x, y) = $0match Some((0, 1)) {
        Some(it) => it,
        None => return,
    };
}
"#,
            r#"
fn f() {
    let Some((x, y)) = Some((0, 1)) else { return };
}
"#,
        );
    }

    #[test]
    fn diverging_block() {
        check_assist(
            convert_match_to_let_else,
            r#"
//- minicore: option
fn f() {
    let x = $0match Some(()) {
        Some(it) => it,
        None => {//comment
            println!("nope");
            return
        },
    };
}
"#,
            r#"
fn f() {
    let Some(x) = Some(()) else {//comment
            println!("nope");
            return
        };
}
"#,
        );
    }
}
