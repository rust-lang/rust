use hir::known;
use ide_db::famous_defs::FamousDefs;
use stdx::format_to;
use syntax::{
    ast::{self, edit_in_place::Indent, make, HasArgList, HasLoopBody},
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
pub(crate) fn convert_iter_for_each_to_for(
    acc: &mut Assists,
    ctx: &AssistContext<'_>,
) -> Option<()> {
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
    let range = stmt.as_ref().map_or(method.syntax(), AstNode::syntax).text_range();

    acc.add(
        AssistId("convert_iter_for_each_to_for", AssistKind::RefactorRewrite),
        "Replace this `Iterator::for_each` with a for loop",
        range,
        |builder| {
            let indent =
                stmt.as_ref().map_or_else(|| method.indent_level(), ast::ExprStmt::indent_level);

            let block = match body {
                ast::Expr::BlockExpr(block) => block,
                _ => make::block_expr(Vec::new(), Some(body)),
            }
            .clone_for_update();
            block.reindent_to(indent);

            let expr_for_loop = make::expr_for_loop(param, receiver, block);
            builder.replace(range, expr_for_loop.to_string())
        },
    )
}

// Assist: convert_for_loop_with_for_each
//
// Converts a for loop into a for_each loop on the Iterator.
//
// ```
// fn main() {
//     let x = vec![1, 2, 3];
//     for$0 v in x {
//         let y = v * 2;
//     }
// }
// ```
// ->
// ```
// fn main() {
//     let x = vec![1, 2, 3];
//     x.into_iter().for_each(|v| {
//         let y = v * 2;
//     });
// }
// ```
pub(crate) fn convert_for_loop_with_for_each(
    acc: &mut Assists,
    ctx: &AssistContext<'_>,
) -> Option<()> {
    let for_loop = ctx.find_node_at_offset::<ast::ForExpr>()?;
    let iterable = for_loop.iterable()?;
    let pat = for_loop.pat()?;
    let body = for_loop.loop_body()?;
    if body.syntax().text_range().start() < ctx.offset() {
        cov_mark::hit!(not_available_in_body);
        return None;
    }

    acc.add(
        AssistId("convert_for_loop_with_for_each", AssistKind::RefactorRewrite),
        "Replace this for loop with `Iterator::for_each`",
        for_loop.syntax().text_range(),
        |builder| {
            let mut buf = String::new();

            if let Some((expr_behind_ref, method)) =
                is_ref_and_impls_iter_method(&ctx.sema, &iterable)
            {
                // We have either "for x in &col" and col implements a method called iter
                //             or "for x in &mut col" and col implements a method called iter_mut
                format_to!(buf, "{expr_behind_ref}.{method}()");
            } else if let ast::Expr::RangeExpr(..) = iterable {
                // range expressions need to be parenthesized for the syntax to be correct
                format_to!(buf, "({iterable})");
            } else if impls_core_iter(&ctx.sema, &iterable) {
                format_to!(buf, "{iterable}");
            } else if let ast::Expr::RefExpr(_) = iterable {
                format_to!(buf, "({iterable}).into_iter()");
            } else {
                format_to!(buf, "{iterable}.into_iter()");
            }

            format_to!(buf, ".for_each(|{pat}| {body});");

            builder.replace(for_loop.syntax().text_range(), buf)
        },
    )
}

/// If iterable is a reference where the expression behind the reference implements a method
/// returning an Iterator called iter or iter_mut (depending on the type of reference) then return
/// the expression behind the reference and the method name
fn is_ref_and_impls_iter_method(
    sema: &hir::Semantics<'_, ide_db::RootDatabase>,
    iterable: &ast::Expr,
) -> Option<(ast::Expr, hir::Name)> {
    let ref_expr = match iterable {
        ast::Expr::RefExpr(r) => r,
        _ => return None,
    };
    let wanted_method = if ref_expr.mut_token().is_some() { known::iter_mut } else { known::iter };
    let expr_behind_ref = ref_expr.expr()?;
    let ty = sema.type_of_expr(&expr_behind_ref)?.adjusted();
    let scope = sema.scope(iterable.syntax())?;
    let krate = scope.krate();
    let iter_trait = FamousDefs(sema, krate).core_iter_Iterator()?;

    let has_wanted_method = ty
        .iterate_method_candidates(sema.db, &scope, None, Some(&wanted_method), |func| {
            if func.ret_type(sema.db).impls_trait(sema.db, iter_trait, &[]) {
                return Some(());
            }
            None
        })
        .is_some();
    if !has_wanted_method {
        return None;
    }

    Some((expr_behind_ref, wanted_method))
}

/// Whether iterable implements core::Iterator
fn impls_core_iter(sema: &hir::Semantics<'_, ide_db::RootDatabase>, iterable: &ast::Expr) -> bool {
    (|| {
        let it_typ = sema.type_of_expr(iterable)?.adjusted();

        let module = sema.scope(iterable.syntax())?.module();

        let krate = module.krate();
        let iter_trait = FamousDefs(sema, krate).core_iter_Iterator()?;
        cov_mark::hit!(test_already_impls_iterator);
        Some(it_typ.impls_trait(sema.db, iter_trait, &[]))
    })()
    .unwrap_or(false)
}

fn validate_method_call_expr(
    ctx: &AssistContext<'_>,
    expr: ast::MethodCallExpr,
) -> Option<(ast::Expr, ast::Expr)> {
    let name_ref = expr.name_ref()?;
    if !name_ref.syntax().text_range().contains_range(ctx.selection_trimmed()) {
        cov_mark::hit!(test_for_each_not_applicable_invalid_cursor_pos);
        return None;
    }
    if name_ref.text() != "for_each" {
        return None;
    }

    let sema = &ctx.sema;

    let receiver = expr.receiver()?;
    let expr = ast::Expr::MethodCallExpr(expr);

    let it_type = sema.type_of_expr(&receiver)?.adjusted();
    let module = sema.scope(receiver.syntax())?.module();
    let krate = module.krate();

    let iter_trait = FamousDefs(sema, krate).core_iter_Iterator()?;
    it_type.impls_trait(sema.db, iter_trait, &[]).then_some((expr, receiver))
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

    #[test]
    fn each_to_for_not_for() {
        check_assist_not_applicable(
            convert_for_loop_with_for_each,
            r"
let mut x = vec![1, 2, 3];
x.iter_mut().$0for_each(|v| *v *= 2);
        ",
        )
    }

    #[test]
    fn each_to_for_simple_for() {
        check_assist(
            convert_for_loop_with_for_each,
            r"
fn main() {
    let x = vec![1, 2, 3];
    for $0v in x {
        v *= 2;
    }
}",
            r"
fn main() {
    let x = vec![1, 2, 3];
    x.into_iter().for_each(|v| {
        v *= 2;
    });
}",
        )
    }

    #[test]
    fn each_to_for_for_in_range() {
        check_assist(
            convert_for_loop_with_for_each,
            r#"
//- minicore: range, iterators
impl<T> core::iter::Iterator for core::ops::Range<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        None
    }
}

fn main() {
    for $0x in 0..92 {
        print!("{}", x);
    }
}"#,
            r#"
impl<T> core::iter::Iterator for core::ops::Range<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        None
    }
}

fn main() {
    (0..92).for_each(|x| {
        print!("{}", x);
    });
}"#,
        )
    }

    #[test]
    fn each_to_for_not_available_in_body() {
        cov_mark::check!(not_available_in_body);
        check_assist_not_applicable(
            convert_for_loop_with_for_each,
            r"
fn main() {
    let x = vec![1, 2, 3];
    for v in x {
        $0v *= 2;
    }
}",
        )
    }

    #[test]
    fn each_to_for_for_borrowed() {
        check_assist(
            convert_for_loop_with_for_each,
            r#"
//- minicore: iterators
use core::iter::{Repeat, repeat};

struct S;
impl S {
    fn iter(&self) -> Repeat<i32> { repeat(92) }
    fn iter_mut(&mut self) -> Repeat<i32> { repeat(92) }
}

fn main() {
    let x = S;
    for $0v in &x {
        let a = v * 2;
    }
}
"#,
            r#"
use core::iter::{Repeat, repeat};

struct S;
impl S {
    fn iter(&self) -> Repeat<i32> { repeat(92) }
    fn iter_mut(&mut self) -> Repeat<i32> { repeat(92) }
}

fn main() {
    let x = S;
    x.iter().for_each(|v| {
        let a = v * 2;
    });
}
"#,
        )
    }

    #[test]
    fn each_to_for_for_borrowed_no_iter_method() {
        check_assist(
            convert_for_loop_with_for_each,
            r"
struct NoIterMethod;
fn main() {
    let x = NoIterMethod;
    for $0v in &x {
        let a = v * 2;
    }
}
",
            r"
struct NoIterMethod;
fn main() {
    let x = NoIterMethod;
    (&x).into_iter().for_each(|v| {
        let a = v * 2;
    });
}
",
        )
    }

    #[test]
    fn each_to_for_for_borrowed_mut() {
        check_assist(
            convert_for_loop_with_for_each,
            r#"
//- minicore: iterators
use core::iter::{Repeat, repeat};

struct S;
impl S {
    fn iter(&self) -> Repeat<i32> { repeat(92) }
    fn iter_mut(&mut self) -> Repeat<i32> { repeat(92) }
}

fn main() {
    let x = S;
    for $0v in &mut x {
        let a = v * 2;
    }
}
"#,
            r#"
use core::iter::{Repeat, repeat};

struct S;
impl S {
    fn iter(&self) -> Repeat<i32> { repeat(92) }
    fn iter_mut(&mut self) -> Repeat<i32> { repeat(92) }
}

fn main() {
    let x = S;
    x.iter_mut().for_each(|v| {
        let a = v * 2;
    });
}
"#,
        )
    }

    #[test]
    fn each_to_for_for_borrowed_mut_behind_var() {
        check_assist(
            convert_for_loop_with_for_each,
            r"
fn main() {
    let x = vec![1, 2, 3];
    let y = &mut x;
    for $0v in y {
        *v *= 2;
    }
}",
            r"
fn main() {
    let x = vec![1, 2, 3];
    let y = &mut x;
    y.into_iter().for_each(|v| {
        *v *= 2;
    });
}",
        )
    }

    #[test]
    fn each_to_for_already_impls_iterator() {
        cov_mark::check!(test_already_impls_iterator);
        check_assist(
            convert_for_loop_with_for_each,
            r#"
//- minicore: iterators
fn main() {
    for$0 a in core::iter::repeat(92).take(1) {
        println!("{}", a);
    }
}
"#,
            r#"
fn main() {
    core::iter::repeat(92).take(1).for_each(|a| {
        println!("{}", a);
    });
}
"#,
        );
    }
}
