use ide_db::helpers::FamousDefs;
use syntax::{
    ast::{self, edit::AstNodeEdit, make, ArgListOwner},
    AstNode,
};

use crate::{AssistContext, AssistId, AssistKind, Assists};

// Assist: convert_iter_for_each_to_for
//
// Converts an Iterator::for_each function into a for loop.
//
// ```
// # //- minicore: iterators
// # use core::iter;
// fn main() {
//     let iter = iter::repeat((9, 2));
//     iter.for_each$0(|(x, y)| {
//         println!("x: {}, y: {}", x, y);
//     });
// }
// ```
// ->
// ```
// # use core::iter;
// fn main() {
//     let iter = iter::repeat((9, 2));
//     for (x, y) in iter {
//         println!("x: {}, y: {}", x, y);
//     }
// }
// ```

pub(crate) fn convert_iter_for_each_to_for(acc: &mut Assists, ctx: &AssistContext) -> Option<()> {
    let method = ctx.find_node_at_offset::<ast::MethodCallExpr>()?;

    let closure = match method.arg_list()?.args().next()? {
        ast::Expr::ClosureExpr(expr) => expr,
        _ => return None,
    };

    let (method, receiver) = validate_method_call_expr(ctx, method)?;

    let param_list = closure.param_list()?;
    let param = param_list.params().next()?.pat()?;
    let body = closure.body()?;

    let stmt = method.syntax().parent().and_then(ast::ExprStmt::cast);
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
    ctx: &AssistContext,
    expr: ast::MethodCallExpr,
) -> Option<(ast::Expr, ast::Expr)> {
    let name_ref = expr.name_ref()?;
    if name_ref.syntax().text_range().intersect(ctx.frange.range).is_none() {
        cov_mark::hit!(test_for_each_not_applicable_invalid_cursor_pos);
        return None;
    }
    if name_ref.text() != "for_each" {
        return None;
    }

    let sema = &ctx.sema;

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

    #[test]
    fn test_for_each_in_method_stmt() {
        check_assist(
            convert_iter_for_each_to_for,
            r#"
//- minicore: iterators
fn main() {
    let it = core::iter::repeat(92);
    it.$0for_each(|(x, y)| {
        println!("x: {}, y: {}", x, y);
    });
}
"#,
            r#"
fn main() {
    let it = core::iter::repeat(92);
    for (x, y) in it {
        println!("x: {}, y: {}", x, y);
    }
}
"#,
        )
    }

    #[test]
    fn test_for_each_in_method() {
        check_assist(
            convert_iter_for_each_to_for,
            r#"
//- minicore: iterators
fn main() {
    let it = core::iter::repeat(92);
    it.$0for_each(|(x, y)| {
        println!("x: {}, y: {}", x, y);
    })
}
"#,
            r#"
fn main() {
    let it = core::iter::repeat(92);
    for (x, y) in it {
        println!("x: {}, y: {}", x, y);
    }
}
"#,
        )
    }

    #[test]
    fn test_for_each_without_braces_stmt() {
        check_assist(
            convert_iter_for_each_to_for,
            r#"
//- minicore: iterators
fn main() {
    let it = core::iter::repeat(92);
    it.$0for_each(|(x, y)| println!("x: {}, y: {}", x, y));
}
"#,
            r#"
fn main() {
    let it = core::iter::repeat(92);
    for (x, y) in it {
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
//- minicore: iterators
fn main() {
    ().$0for_each(|x| println!("{}", x));
}"#,
        )
    }

    #[test]
    fn test_for_each_not_applicable_invalid_cursor_pos() {
        cov_mark::check!(test_for_each_not_applicable_invalid_cursor_pos);
        check_assist_not_applicable(
            convert_iter_for_each_to_for,
            r#"
//- minicore: iterators
fn main() {
    core::iter::repeat(92).for_each(|(x, y)| $0println!("x: {}, y: {}", x, y));
}"#,
        )
    }
}
