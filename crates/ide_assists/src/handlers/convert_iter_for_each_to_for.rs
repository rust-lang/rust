use ide_db::helpers::FamousDefs;
use syntax::{AstNode, ast::{self, make, ArgListOwner, edit::AstNodeEdit}};

use crate::{AssistContext, AssistId, AssistKind, Assists};

/// Assist: convert_iter_for_each_to_for
//
/// Converts an Iterator::for_each function into a for loop.
///
/// ```rust
/// fn main() {
///     let vec = vec![(1, 2), (2, 3), (3, 4)];
///     x.iter().for_each(|(x, y)| {
///         println!("x: {}, y: {}", x, y);
///    })
/// }
/// ```
/// ->
/// ```rust
/// fn main() {
///     let vec = vec![(1, 2), (2, 3), (3, 4)];
///     for (x, y) in x.iter() {
///         println!("x: {}, y: {}", x, y);
///     });
/// }
/// ```
pub(crate) fn convert_iter_for_each_to_for(acc: &mut Assists, ctx: &AssistContext) -> Option<()> {
    let closure;

    let total_expr = match ctx.find_node_at_offset::<ast::Expr>()? {
        ast::Expr::MethodCallExpr(expr) => {
            closure = match expr.arg_list()?.args().next()? {
                ast::Expr::ClosureExpr(expr) => expr,
                _ => { return None; }
            };
            
            expr
        },
        ast::Expr::ClosureExpr(expr) => {
            closure = expr;
            ast::MethodCallExpr::cast(closure.syntax().ancestors().nth(2)?)?
        },
        _ => { return None; }
    };

    let (total_expr, parent) = validate_method_call_expr(&ctx.sema, total_expr)?;

    let param_list = closure.param_list()?;
    let param = param_list.params().next()?.pat()?;
    let body = closure.body()?;

    acc.add(
        AssistId("convert_iter_for_each_to_for", AssistKind::RefactorRewrite),
        "Replace this `Iterator::for_each` with a for loop",
        total_expr.syntax().text_range(),
        |builder| {
            let original_indentation = total_expr.indent_level();

            let block = match body {
                ast::Expr::BlockExpr(block) => block,
                _ => make::block_expr(Vec::new(), Some(body))
            }.reset_indent().indent(original_indentation);

            let expr_for_loop = make::expr_for_loop(param, parent, block);
            builder.replace_ast(total_expr, expr_for_loop)
        },
    )
}

fn validate_method_call_expr(
    sema: &hir::Semantics<ide_db::RootDatabase>,
    expr: ast::MethodCallExpr,
) -> Option<(ast::Expr, ast::Expr)> {
    if expr.name_ref()?.text() != "for_each" {
        return None;
    }

    let expr = ast::Expr::MethodCallExpr(expr);
    let parent = ast::Expr::cast(expr.syntax().first_child()?)?;

    let it_type = sema.type_of_expr(&parent)?;
    let module = sema.scope(parent.syntax()).module()?;
    let krate = module.krate();

    let iter_trait = FamousDefs(sema, Some(krate)).core_iter_Iterator()?;
    it_type.impls_trait(sema.db, iter_trait, &[]).then(|| (expr, parent))
}

#[cfg(test)]
mod tests {
    use crate::tests::check_assist;

    use super::*;

    fn check_assist_with_fixtures(before: &str, after: &str) {
        let before = &format!(
            "//- /main.rs crate:main deps:core,empty_iter{}{}",
            before,
            FamousDefs::FIXTURE,
        );
        check_assist(convert_iter_for_each_to_for, before, after);
    }

    #[test]
    fn test_for_each_in_method() {
        check_assist_with_fixtures(
            r#"
fn main() {
    let x = vec![(1, 1), (2, 2), (3, 3), (4, 4)];
    x.iter().$0for_each(|(x, y)| {
        println!("x: {}, y: {}", x, y);
    });
}"#,
            r#"
fn main() {
    let x = vec![(1, 1), (2, 2), (3, 3), (4, 4)];
    for (x, y) in x.iter() {
        println!("x: {}, y: {}", x, y);
    };
}"#,
        )
    }

    #[test]
    fn test_for_each_without_braces() {
        check_assist_with_fixtures(
            r#"
fn main() {
    let x = vec![(1, 1), (2, 2), (3, 3), (4, 4)];
    x.iter().$0for_each(|(x, y)| println!("x: {}, y: {}", x, y));
}"#,
            r#"
fn main() {
    let x = vec![(1, 1), (2, 2), (3, 3), (4, 4)];
    for (x, y) in x.iter() {
        println!("x: {}, y: {}", x, y)
    };
}"#,
        )
    }

    #[test]
    fn test_for_each_in_closure() {
        check_assist_with_fixtures(
            r#"
fn main() {
    let x = vec![(1, 1), (2, 2), (3, 3), (4, 4)];
    x.iter().for_each($0|(x, y)| println!("x: {}, y: {}", x, y));
}"#,
            r#"
fn main() {
    let x = vec![(1, 1), (2, 2), (3, 3), (4, 4)];
    for (x, y) in x.iter() {
        println!("x: {}, y: {}", x, y)
    };
}"#,
        )
    }
}