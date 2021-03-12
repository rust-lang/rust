use ide_db::helpers::FamousDefs;
use syntax::{
    ast::{self, edit::AstNodeEdit, make, ArgListOwner},
    AstNode,
};

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
    let method = ctx.find_node_at_offset::<ast::MethodCallExpr>()?;
    let stmt = method.syntax().parent().and_then(ast::ExprStmt::cast);

    let closure = match method.arg_list()?.args().next()? {
        ast::Expr::ClosureExpr(expr) => expr,
        _ => return None,
    };

    let (method, receiver) = validate_method_call_expr(&ctx.sema, method)?;

    let param_list = closure.param_list()?;
    let param = param_list.params().next()?.pat()?;
    let body = closure.body()?;

    let syntax = stmt.as_ref().map_or(method.syntax(), |stmt| stmt.syntax());

    acc.add(
        AssistId("convert_iter_for_each_to_for", AssistKind::RefactorRewrite),
        "Replace this `Iterator::for_each` with a for loop",
        syntax.text_range(),
        |builder| {
            let indent = stmt.as_ref().map_or(method.indent_level(), |stmt| stmt.indent_level());

            let block = match body {
                ast::Expr::BlockExpr(block) => block,
                _ => make::block_expr(Vec::new(), Some(body)),
            }
            .reset_indent()
            .indent(indent);

            let expr_for_loop = make::expr_for_loop(param, receiver, block);
            builder.replace(syntax.text_range(), expr_for_loop.syntax().text())
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

    let receiver = expr.receiver()?;
    let expr = ast::Expr::MethodCallExpr(expr);

    let it_type = sema.type_of_expr(&receiver)?;
    let module = sema.scope(receiver.syntax()).module()?;
    let krate = module.krate();

    let iter_trait = FamousDefs(sema, Some(krate)).core_iter_Iterator()?;
    it_type.impls_trait(sema.db, iter_trait, &[]).then(|| (expr, receiver))
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_assist, check_assist_not_applicable};

    use super::*;

    const EMPTY_ITER_FIXTURE: &'static str = r"
//- /lib.rs deps:core crate:empty_iter
pub struct EmptyIter;
impl Iterator for EmptyIter {
    type Item = usize;
    fn next(&mut self) -> Option<Self::Item> { None }
}
pub struct Empty;
impl Empty {
    pub fn iter(&self) -> EmptyIter { EmptyIter }
}
";

    fn check_assist_with_fixtures(before: &str, after: &str) {
        let before = &format!(
            "//- /main.rs crate:main deps:core,empty_iter{}{}{}",
            before,
            EMPTY_ITER_FIXTURE,
            FamousDefs::FIXTURE,
        );
        check_assist(convert_iter_for_each_to_for, before, after);
    }

    #[test]
    fn test_for_each_in_method_stmt() {
        check_assist_with_fixtures(
            r#"
use empty_iter::*;
fn main() {
    let x = Empty;
    x.iter().$0for_each(|(x, y)| {
        println!("x: {}, y: {}", x, y);
    });
}"#,
            r#"
use empty_iter::*;
fn main() {
    let x = Empty;
    for (x, y) in x.iter() {
        println!("x: {}, y: {}", x, y);
    }
}
"#,
        )
    }

    #[test]
    fn test_for_each_in_method() {
        check_assist_with_fixtures(
            r#"
use empty_iter::*;
fn main() {
    let x = Empty;
    x.iter().$0for_each(|(x, y)| {
        println!("x: {}, y: {}", x, y);
    })
}"#,
            r#"
use empty_iter::*;
fn main() {
    let x = Empty;
    for (x, y) in x.iter() {
        println!("x: {}, y: {}", x, y);
    }
}
"#,
        )
    }

    #[test]
    fn test_for_each_in_iter_stmt() {
        check_assist_with_fixtures(
            r#"
use empty_iter::*;
fn main() {
    let x = Empty.iter();
    x.$0for_each(|(x, y)| {
        println!("x: {}, y: {}", x, y);
    });
}"#,
            r#"
use empty_iter::*;
fn main() {
    let x = Empty.iter();
    for (x, y) in x {
        println!("x: {}, y: {}", x, y);
    }
}
"#,
        )
    }

    #[test]
    fn test_for_each_without_braces_stmt() {
        check_assist_with_fixtures(
            r#"
use empty_iter::*;
fn main() {
    let x = Empty;
    x.iter().$0for_each(|(x, y)| println!("x: {}, y: {}", x, y));
}"#,
            r#"
use empty_iter::*;
fn main() {
    let x = Empty;
    for (x, y) in x.iter() {
        println!("x: {}, y: {}", x, y)
    }
}
"#,
        )
    }
    #[test]
    fn test_for_each_not_applicable() {
        check_assist_not_applicable(
            convert_iter_for_each_to_for,
            r#"
fn main() {
    value.$0for_each(|x| println!("{}", x));
}"#,
        )
    }
}
