use ast::LoopBodyOwner;
use hir::known;
use ide_db::helpers::FamousDefs;
use stdx::format_to;
use syntax::{ast, AstNode};

use crate::{AssistContext, AssistId, AssistKind, Assists};

// Assist: replace_for_loop_with_for_each
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
pub(crate) fn replace_for_loop_with_for_each(acc: &mut Assists, ctx: &AssistContext) -> Option<()> {
    let for_loop = ctx.find_node_at_offset::<ast::ForExpr>()?;
    let iterable = for_loop.iterable()?;
    let pat = for_loop.pat()?;
    let body = for_loop.loop_body()?;
    if body.syntax().text_range().start() < ctx.offset() {
        cov_mark::hit!(not_available_in_body);
        return None;
    }

    acc.add(
        AssistId("replace_for_loop_with_for_each", AssistKind::RefactorRewrite),
        "Replace this for loop with `Iterator::for_each`",
        for_loop.syntax().text_range(),
        |builder| {
            let mut buf = String::new();

            if let Some((expr_behind_ref, method)) =
                is_ref_and_impls_iter_method(&ctx.sema, &iterable)
            {
                // We have either "for x in &col" and col implements a method called iter
                //             or "for x in &mut col" and col implements a method called iter_mut
                format_to!(buf, "{}.{}()", expr_behind_ref, method);
            } else if impls_core_iter(&ctx.sema, &iterable) {
                format_to!(buf, "{}", iterable);
            } else {
                if let ast::Expr::RefExpr(_) = iterable {
                    format_to!(buf, "({}).into_iter()", iterable);
                } else {
                    format_to!(buf, "{}.into_iter()", iterable);
                }
            }

            format_to!(buf, ".for_each(|{}| {});", pat, body);

            builder.replace(for_loop.syntax().text_range(), buf)
        },
    )
}

/// If iterable is a reference where the expression behind the reference implements a method
/// returning an Iterator called iter or iter_mut (depending on the type of reference) then return
/// the expression behind the reference and the method name
fn is_ref_and_impls_iter_method(
    sema: &hir::Semantics<ide_db::RootDatabase>,
    iterable: &ast::Expr,
) -> Option<(ast::Expr, hir::Name)> {
    let ref_expr = match iterable {
        ast::Expr::RefExpr(r) => r,
        _ => return None,
    };
    let wanted_method = if ref_expr.mut_token().is_some() { known::iter_mut } else { known::iter };
    let expr_behind_ref = ref_expr.expr()?;
    let typ = sema.type_of_expr(&expr_behind_ref)?;
    let scope = sema.scope(iterable.syntax());
    let krate = scope.module()?.krate();
    let traits_in_scope = scope.traits_in_scope();
    let iter_trait = FamousDefs(sema, Some(krate)).core_iter_Iterator()?;
    let has_wanted_method = typ.iterate_method_candidates(
        sema.db,
        krate,
        &traits_in_scope,
        Some(&wanted_method),
        |_, func| {
            if func.ret_type(sema.db).impls_trait(sema.db, iter_trait, &[]) {
                return Some(());
            }
            None
        },
    );
    has_wanted_method.and(Some((expr_behind_ref, wanted_method)))
}

/// Whether iterable implements core::Iterator
fn impls_core_iter(sema: &hir::Semantics<ide_db::RootDatabase>, iterable: &ast::Expr) -> bool {
    let it_typ = if let Some(i) = sema.type_of_expr(iterable) {
        i
    } else {
        return false;
    };
    let module = if let Some(m) = sema.scope(iterable.syntax()).module() {
        m
    } else {
        return false;
    };
    let krate = module.krate();
    if let Some(iter_trait) = FamousDefs(sema, Some(krate)).core_iter_Iterator() {
        return it_typ.impls_trait(sema.db, iter_trait, &[]);
    }
    false
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
    pub fn iter_mut(&self) -> EmptyIter { EmptyIter }
}

pub struct NoIterMethod;
";

    fn check_assist_with_fixtures(before: &str, after: &str) {
        let before = &format!(
            "//- /main.rs crate:main deps:core,empty_iter{}{}{}",
            before,
            FamousDefs::FIXTURE,
            EMPTY_ITER_FIXTURE
        );
        check_assist(replace_for_loop_with_for_each, before, after);
    }

    #[test]
    fn test_not_for() {
        check_assist_not_applicable(
            replace_for_loop_with_for_each,
            r"
let mut x = vec![1, 2, 3];
x.iter_mut().$0for_each(|v| *v *= 2);
        ",
        )
    }

    #[test]
    fn test_simple_for() {
        check_assist(
            replace_for_loop_with_for_each,
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
    fn not_available_in_body() {
        cov_mark::check!(not_available_in_body);
        check_assist_not_applicable(
            replace_for_loop_with_for_each,
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
    fn test_for_borrowed() {
        check_assist_with_fixtures(
            r"
use empty_iter::*;
fn main() {
    let x = Empty;
    for $0v in &x {
        let a = v * 2;
    }
}
",
            r"
use empty_iter::*;
fn main() {
    let x = Empty;
    x.iter().for_each(|v| {
        let a = v * 2;
    });
}
",
        )
    }

    #[test]
    fn test_for_borrowed_no_iter_method() {
        check_assist_with_fixtures(
            r"
use empty_iter::*;
fn main() {
    let x = NoIterMethod;
    for $0v in &x {
        let a = v * 2;
    }
}
",
            r"
use empty_iter::*;
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
    fn test_for_borrowed_mut() {
        check_assist_with_fixtures(
            r"
use empty_iter::*;
fn main() {
    let x = Empty;
    for $0v in &mut x {
        let a = v * 2;
    }
}
",
            r"
use empty_iter::*;
fn main() {
    let x = Empty;
    x.iter_mut().for_each(|v| {
        let a = v * 2;
    });
}
",
        )
    }

    #[test]
    fn test_for_borrowed_mut_behind_var() {
        check_assist(
            replace_for_loop_with_for_each,
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
    fn test_already_impls_iterator() {
        check_assist_with_fixtures(
            r#"
use empty_iter::*;
fn main() {
    let x = Empty;
    for$0 a in x.iter().take(1) {
        println!("{}", a);
    }
}
"#,
            r#"
use empty_iter::*;
fn main() {
    let x = Empty;
    x.iter().take(1).for_each(|a| {
        println!("{}", a);
    });
}
"#,
        );
    }
}
