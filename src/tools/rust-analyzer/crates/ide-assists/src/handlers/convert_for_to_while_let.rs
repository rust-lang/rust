use hir::{
    Name,
    sym::{self},
};
use ide_db::{famous_defs::FamousDefs, syntax_helpers::suggest_name};
use syntax::{
    AstNode,
    ast::{self, HasLoopBody, edit::IndentLevel, make, syntax_factory::SyntaxFactory},
    syntax_editor::Position,
};

use crate::{AssistContext, AssistId, Assists};

// Assist: convert_for_loop_to_while_let
//
// Converts a for loop into a while let on the Iterator.
//
// ```
// fn main() {
//     let x = vec![1, 2, 3];
//     for$0 v in x {
//         let y = v * 2;
//     };
// }
// ```
// ->
// ```
// fn main() {
//     let x = vec![1, 2, 3];
//     let mut tmp = x.into_iter();
//     while let Some(v) = tmp.next() {
//         let y = v * 2;
//     };
// }
// ```
pub(crate) fn convert_for_loop_to_while_let(
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
        AssistId::refactor_rewrite("convert_for_loop_to_while_let"),
        "Replace this for loop with `while let`",
        for_loop.syntax().text_range(),
        |builder| {
            let make = SyntaxFactory::with_mappings();
            let mut editor = builder.make_editor(for_loop.syntax());

            let (iterable, method) = if impls_core_iter(&ctx.sema, &iterable) {
                (iterable, None)
            } else if let Some((expr, method)) = is_ref_and_impls_iter_method(&ctx.sema, &iterable)
            {
                (expr, Some(make.name_ref(method.as_str())))
            } else if let ast::Expr::RefExpr(_) = iterable {
                (make::expr_paren(iterable).into(), Some(make.name_ref("into_iter")))
            } else {
                (iterable, Some(make.name_ref("into_iter")))
            };

            let iterable = if let Some(method) = method {
                make::expr_method_call(iterable, method, make::arg_list([])).into()
            } else {
                iterable
            };

            let mut new_name = suggest_name::NameGenerator::new_from_scope_locals(
                ctx.sema.scope(for_loop.syntax()),
            );
            let tmp_var = new_name.suggest_name("tmp");

            let mut_expr = make.let_stmt(
                make.ident_pat(false, true, make.name(&tmp_var)).into(),
                None,
                Some(iterable),
            );
            let indent = IndentLevel::from_node(for_loop.syntax());
            editor.insert(
                Position::before(for_loop.syntax()),
                make::tokens::whitespace(format!("\n{indent}").as_str()),
            );
            editor.insert(Position::before(for_loop.syntax()), mut_expr.syntax());

            let opt_pat = make.tuple_struct_pat(make::ext::ident_path("Some"), [pat]);
            let iter_next_expr = make.expr_method_call(
                make.expr_path(make::ext::ident_path(&tmp_var)),
                make.name_ref("next"),
                make.arg_list([]),
            );
            let cond = make.expr_let(opt_pat.into(), iter_next_expr.into());

            let while_loop = make.expr_while_loop(cond.into(), body);

            editor.replace(for_loop.syntax(), while_loop.syntax());

            editor.add_mappings(make.finish_with_mappings());
            builder.add_file_edits(ctx.vfs_file_id(), editor);
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
    let wanted_method = Name::new_symbol_root(if ref_expr.mut_token().is_some() {
        sym::iter_mut
    } else {
        sym::iter
    });
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

#[cfg(test)]
mod tests {
    use crate::tests::{check_assist, check_assist_not_applicable};

    use super::*;

    #[test]
    fn each_to_for_simple_for() {
        check_assist(
            convert_for_loop_to_while_let,
            r"
fn main() {
    let mut x = vec![1, 2, 3];
    for $0v in x {
        v *= 2;
    };
}",
            r"
fn main() {
    let mut x = vec![1, 2, 3];
    let mut tmp = x.into_iter();
    while let Some(v) = tmp.next() {
        v *= 2;
    };
}",
        )
    }

    #[test]
    fn each_to_for_for_in_range() {
        check_assist(
            convert_for_loop_to_while_let,
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
    let mut tmp = 0..92;
    while let Some(x) = tmp.next() {
        print!("{}", x);
    }
}"#,
        )
    }

    #[test]
    fn each_to_for_not_available_in_body() {
        cov_mark::check!(not_available_in_body);
        check_assist_not_applicable(
            convert_for_loop_to_while_let,
            r"
fn main() {
    let mut x = vec![1, 2, 3];
    for v in x {
        $0v *= 2;
    }
}",
        )
    }

    #[test]
    fn each_to_for_for_borrowed() {
        check_assist(
            convert_for_loop_to_while_let,
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
    let mut tmp = x.iter();
    while let Some(v) = tmp.next() {
        let a = v * 2;
    }
}
"#,
        )
    }

    #[test]
    fn each_to_for_for_borrowed_no_iter_method() {
        check_assist(
            convert_for_loop_to_while_let,
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
    let mut tmp = (&x).into_iter();
    while let Some(v) = tmp.next() {
        let a = v * 2;
    }
}
",
        )
    }

    #[test]
    fn each_to_for_for_borrowed_no_iter_method_mut() {
        check_assist(
            convert_for_loop_to_while_let,
            r"
struct NoIterMethod;
fn main() {
    let x = NoIterMethod;
    for $0v in &mut x {
        let a = v * 2;
    }
}
",
            r"
struct NoIterMethod;
fn main() {
    let x = NoIterMethod;
    let mut tmp = (&mut x).into_iter();
    while let Some(v) = tmp.next() {
        let a = v * 2;
    }
}
",
        )
    }

    #[test]
    fn each_to_for_for_borrowed_mut() {
        check_assist(
            convert_for_loop_to_while_let,
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
    let mut tmp = x.iter_mut();
    while let Some(v) = tmp.next() {
        let a = v * 2;
    }
}
"#,
        )
    }

    #[test]
    fn each_to_for_for_borrowed_mut_behind_var() {
        check_assist(
            convert_for_loop_to_while_let,
            r"
fn main() {
    let mut x = vec![1, 2, 3];
    let y = &mut x;
    for $0v in y {
        *v *= 2;
    }
}",
            r"
fn main() {
    let mut x = vec![1, 2, 3];
    let y = &mut x;
    let mut tmp = y.into_iter();
    while let Some(v) = tmp.next() {
        *v *= 2;
    }
}",
        )
    }

    #[test]
    fn each_to_for_already_impls_iterator() {
        cov_mark::check!(test_already_impls_iterator);
        check_assist(
            convert_for_loop_to_while_let,
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
    let mut tmp = core::iter::repeat(92).take(1);
    while let Some(a) = tmp.next() {
        println!("{}", a);
    }
}
"#,
        );
    }
}
