use hir::HirDisplay;
use ide_db::syntax_helpers::node_ext::walk_ty;
use syntax::ast::{self, AstNode, LetStmt, Param};

use crate::{AssistContext, AssistId, Assists};

// Assist: add_explicit_type
//
// Specify type for a let binding.
//
// ```
// fn main() {
//     let x$0 = 92;
// }
// ```
// ->
// ```
// fn main() {
//     let x: i32 = 92;
// }
// ```
pub(crate) fn add_explicit_type(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    let (ascribed_ty, expr, pat) = if let Some(let_stmt) = ctx.find_node_at_offset::<LetStmt>() {
        let cursor_in_range = {
            let eq_range = let_stmt.eq_token()?.text_range();
            ctx.offset() < eq_range.start()
        };
        if !cursor_in_range {
            cov_mark::hit!(add_explicit_type_not_applicable_if_cursor_after_equals);
            return None;
        }

        (let_stmt.ty(), let_stmt.initializer(), let_stmt.pat()?)
    } else if let Some(param) = ctx.find_node_at_offset::<Param>() {
        if param.syntax().ancestors().nth(2).and_then(ast::ClosureExpr::cast).is_none() {
            cov_mark::hit!(add_explicit_type_not_applicable_in_fn_param);
            return None;
        }
        (param.ty(), None, param.pat()?)
    } else {
        return None;
    };

    let module = ctx.sema.scope(pat.syntax())?.module();
    let pat_range = pat.syntax().text_range();

    // Don't enable the assist if there is a type ascription without any placeholders
    if let Some(ty) = &ascribed_ty {
        let mut contains_infer_ty = false;
        walk_ty(ty, &mut |ty| {
            contains_infer_ty |= matches!(ty, ast::Type::InferType(_));
            false
        });
        if !contains_infer_ty {
            cov_mark::hit!(add_explicit_type_not_applicable_if_ty_already_specified);
            return None;
        }
    }

    let ty = match (pat, expr) {
        (ast::Pat::IdentPat(_), Some(expr)) => ctx.sema.type_of_expr(&expr)?,
        (pat, _) => ctx.sema.type_of_pat(&pat)?,
    }
    .adjusted();

    // Fully unresolved or unnameable types can't be annotated
    if (ty.contains_unknown() && ty.type_arguments().count() == 0) || ty.is_closure() {
        cov_mark::hit!(add_explicit_type_not_applicable_if_ty_not_inferred);
        return None;
    }

    let inferred_type = ty.display_source_code(ctx.db(), module.into(), false).ok()?;
    acc.add(
        AssistId::refactor_rewrite("add_explicit_type"),
        format!("Insert explicit type `{inferred_type}`"),
        pat_range,
        |builder| match ascribed_ty {
            Some(ascribed_ty) => {
                builder.replace(ascribed_ty.syntax().text_range(), inferred_type);
            }
            None => {
                builder.insert(pat_range.end(), format!(": {inferred_type}"));
            }
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::tests::{check_assist, check_assist_not_applicable, check_assist_target};

    #[test]
    fn add_explicit_type_target() {
        check_assist_target(add_explicit_type, r#"fn f() { let a$0 = 1; }"#, "a");
    }

    #[test]
    fn add_explicit_type_simple() {
        check_assist(
            add_explicit_type,
            r#"fn f() { let a$0 = 1; }"#,
            r#"fn f() { let a: i32 = 1; }"#,
        );
    }

    #[test]
    fn add_explicit_type_simple_on_infer_ty() {
        check_assist(
            add_explicit_type,
            r#"fn f() { let a$0: _ = 1; }"#,
            r#"fn f() { let a: i32 = 1; }"#,
        );
    }

    #[test]
    fn add_explicit_type_simple_nested_infer_ty() {
        check_assist(
            add_explicit_type,
            r#"
//- minicore: option
fn f() {
    let a$0: Option<_> = Option::Some(1);
}
"#,
            r#"
fn f() {
    let a: Option<i32> = Option::Some(1);
}
"#,
        );
    }

    #[test]
    fn add_explicit_type_macro_call_expr() {
        check_assist(
            add_explicit_type,
            r"macro_rules! v { () => {0u64} } fn f() { let a$0 = v!(); }",
            r"macro_rules! v { () => {0u64} } fn f() { let a: u64 = v!(); }",
        );
    }

    #[test]
    fn add_explicit_type_not_applicable_for_fully_unresolved() {
        cov_mark::check!(add_explicit_type_not_applicable_if_ty_not_inferred);
        check_assist_not_applicable(add_explicit_type, r#"fn f() { let a$0 = None; }"#);
    }

    #[test]
    fn add_explicit_type_applicable_for_partially_unresolved() {
        check_assist(
            add_explicit_type,
            r#"
        struct Vec<T, V> { t: T, v: V }
        impl<T> Vec<T, Vec<ZZZ, i32>> {
            fn new() -> Self {
                panic!()
            }
        }
        fn f() { let a$0 = Vec::new(); }"#,
            r#"
        struct Vec<T, V> { t: T, v: V }
        impl<T> Vec<T, Vec<ZZZ, i32>> {
            fn new() -> Self {
                panic!()
            }
        }
        fn f() { let a: Vec<_, Vec<_, i32>> = Vec::new(); }"#,
        );
    }

    #[test]
    fn add_explicit_type_not_applicable_closure_expr() {
        check_assist_not_applicable(add_explicit_type, r#"fn f() { let a$0 = || {}; }"#);
    }

    #[test]
    fn add_explicit_type_not_applicable_ty_already_specified() {
        cov_mark::check!(add_explicit_type_not_applicable_if_ty_already_specified);
        check_assist_not_applicable(add_explicit_type, r#"fn f() { let a$0: i32 = 1; }"#);
    }

    #[test]
    fn add_explicit_type_not_applicable_cursor_after_equals_of_let() {
        cov_mark::check!(add_explicit_type_not_applicable_if_cursor_after_equals);
        check_assist_not_applicable(
            add_explicit_type,
            r#"fn f() {let a =$0 match 1 {2 => 3, 3 => 5};}"#,
        )
    }

    /// https://github.com/rust-lang/rust-analyzer/issues/2922
    #[test]
    fn regression_issue_2922() {
        check_assist(
            add_explicit_type,
            r#"
fn main() {
    let $0v = [0.0; 2];
}
"#,
            r#"
fn main() {
    let v: [f64; 2] = [0.0; 2];
}
"#,
        );
        // note: this may break later if we add more consteval. it just needs to be something that our
        // consteval engine doesn't understand
        check_assist_not_applicable(
            add_explicit_type,
            r#"
fn main() {
    let $0l = [0.0; unresolved_function(5)];
}
"#,
        );
    }

    #[test]
    fn default_generics_should_not_be_added() {
        check_assist(
            add_explicit_type,
            r#"
struct Test<K, T = u8> { k: K, t: T }

fn main() {
    let test$0 = Test { t: 23u8, k: 33 };
}
"#,
            r#"
struct Test<K, T = u8> { k: K, t: T }

fn main() {
    let test: Test<i32> = Test { t: 23u8, k: 33 };
}
"#,
        );
    }

    #[test]
    fn type_should_be_added_after_pattern() {
        // LetStmt = Attr* 'let' Pat (':' Type)? '=' initializer:Expr ';'
        check_assist(
            add_explicit_type,
            r#"
fn main() {
    let $0test @ () = ();
}
"#,
            r#"
fn main() {
    let test @ (): () = ();
}
"#,
        );
    }

    #[test]
    fn add_explicit_type_inserts_coercions() {
        check_assist(
            add_explicit_type,
            r#"
//- minicore: coerce_unsized
fn f() {
    let $0x: *const [_] = &[3];
}
"#,
            r#"
fn f() {
    let x: *const [i32] = &[3];
}
"#,
        );
    }

    #[test]
    fn add_explicit_type_not_applicable_fn_param() {
        cov_mark::check!(add_explicit_type_not_applicable_in_fn_param);
        check_assist_not_applicable(add_explicit_type, r#"fn f(x$0: ()) {}"#);
    }

    #[test]
    fn add_explicit_type_ascribes_closure_param() {
        check_assist(
            add_explicit_type,
            r#"
fn f() {
    |y$0| {
        let x: i32 = y;
    };
}
"#,
            r#"
fn f() {
    |y: i32| {
        let x: i32 = y;
    };
}
"#,
        );
    }

    #[test]
    fn add_explicit_type_ascribes_closure_param_already_ascribed() {
        check_assist(
            add_explicit_type,
            r#"
//- minicore: option
fn f() {
    |mut y$0: Option<_>| {
        y = Some(3);
    };
}
"#,
            r#"
fn f() {
    |mut y: Option<i32>| {
        y = Some(3);
    };
}
"#,
        );
    }
}
