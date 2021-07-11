use hir::HirDisplay;
use syntax::{
    ast::{self, AstNode, LetStmt},
    TextRange,
};

use crate::{AssistContext, AssistId, AssistKind, Assists};

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
pub(crate) fn add_explicit_type(acc: &mut Assists, ctx: &AssistContext) -> Option<()> {
    let let_stmt = ctx.find_node_at_offset::<LetStmt>()?;
    let module = ctx.sema.scope(let_stmt.syntax()).module()?;
    let expr = let_stmt.initializer()?;
    // Must be a binding
    let pat = match let_stmt.pat()? {
        ast::Pat::IdentPat(bind_pat) => bind_pat,
        _ => return None,
    };
    let pat_range = pat.syntax().text_range();

    // Assist should only be applicable if cursor is between 'let' and '='
    let cursor_in_range = {
        let stmt_range = let_stmt.syntax().text_range();
        let eq_range = let_stmt.eq_token()?.text_range();
        let let_range = TextRange::new(stmt_range.start(), eq_range.start());
        let_range.contains_range(ctx.frange.range)
    };
    if !cursor_in_range {
        cov_mark::hit!(add_explicit_type_not_applicable_if_cursor_after_equals);
        return None;
    }

    // Assist not applicable if the type has already been specified
    // and it has no placeholders
    let ascribed_ty = let_stmt.ty();
    if let Some(ty) = &ascribed_ty {
        if ty.syntax().descendants().find_map(ast::InferType::cast).is_none() {
            cov_mark::hit!(add_explicit_type_not_applicable_if_ty_already_specified);
            return None;
        }
    }

    // Infer type
    let (ty, _) = ctx.sema.type_of_expr_with_coercion(&expr)?;
    if ty.contains_unknown() || ty.is_closure() {
        cov_mark::hit!(add_explicit_type_not_applicable_if_ty_not_inferred);
        return None;
    }

    let inferred_type = ty.display_source_code(ctx.db(), module.into()).ok()?;
    acc.add(
        AssistId("add_explicit_type", AssistKind::RefactorRewrite),
        format!("Insert explicit type `{}`", inferred_type),
        pat_range,
        |builder| match ascribed_ty {
            Some(ascribed_ty) => {
                builder.replace(ascribed_ty.syntax().text_range(), inferred_type);
            }
            None => {
                builder.insert(pat_range.end(), format!(": {}", inferred_type));
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
    fn add_explicit_type_works_for_simple_expr() {
        check_assist(
            add_explicit_type,
            r#"fn f() { let a$0 = 1; }"#,
            r#"fn f() { let a: i32 = 1; }"#,
        );
    }

    #[test]
    fn add_explicit_type_works_for_underscore() {
        check_assist(
            add_explicit_type,
            r#"fn f() { let a$0: _ = 1; }"#,
            r#"fn f() { let a: i32 = 1; }"#,
        );
    }

    #[test]
    fn add_explicit_type_works_for_nested_underscore() {
        check_assist(
            add_explicit_type,
            r#"
enum Option<T> { Some(T), None }

fn f() {
    let a$0: Option<_> = Option::Some(1);
}
"#,
            r#"
enum Option<T> { Some(T), None }

fn f() {
    let a: Option<i32> = Option::Some(1);
}
"#,
        );
    }

    #[test]
    fn add_explicit_type_works_for_macro_call() {
        check_assist(
            add_explicit_type,
            r"macro_rules! v { () => {0u64} } fn f() { let a$0 = v!(); }",
            r"macro_rules! v { () => {0u64} } fn f() { let a: u64 = v!(); }",
        );
    }

    #[test]
    fn add_explicit_type_works_for_macro_call_recursive() {
        check_assist(
            add_explicit_type,
            r#"macro_rules! u { () => {0u64} } macro_rules! v { () => {u!()} } fn f() { let a$0 = v!(); }"#,
            r#"macro_rules! u { () => {0u64} } macro_rules! v { () => {u!()} } fn f() { let a: u64 = v!(); }"#,
        );
    }

    #[test]
    fn add_explicit_type_not_applicable_if_ty_not_inferred() {
        cov_mark::check!(add_explicit_type_not_applicable_if_ty_not_inferred);
        check_assist_not_applicable(add_explicit_type, r#"fn f() { let a$0 = None; }"#);
    }

    #[test]
    fn add_explicit_type_not_applicable_if_ty_already_specified() {
        cov_mark::check!(add_explicit_type_not_applicable_if_ty_already_specified);
        check_assist_not_applicable(add_explicit_type, r#"fn f() { let a$0: i32 = 1; }"#);
    }

    #[test]
    fn add_explicit_type_not_applicable_if_specified_ty_is_tuple() {
        check_assist_not_applicable(
            add_explicit_type,
            r#"fn f() { let a$0: (i32, i32) = (3, 4); }"#,
        );
    }

    #[test]
    fn add_explicit_type_not_applicable_if_cursor_after_equals() {
        cov_mark::check!(add_explicit_type_not_applicable_if_cursor_after_equals);
        check_assist_not_applicable(
            add_explicit_type,
            r#"fn f() {let a =$0 match 1 {2 => 3, 3 => 5};}"#,
        )
    }

    #[test]
    fn add_explicit_type_not_applicable_if_cursor_before_let() {
        check_assist_not_applicable(
            add_explicit_type,
            r#"fn f() $0{let a = match 1 {2 => 3, 3 => 5};}"#,
        )
    }

    #[test]
    fn closure_parameters_are_not_added() {
        check_assist_not_applicable(
            add_explicit_type,
            r#"
fn main() {
    let multiply_by_two$0 = |i| i * 3;
    let six = multiply_by_two(2);
}
"#,
        )
    }

    /// https://github.com/rust-analyzer/rust-analyzer/issues/2922
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
    let $0l = [0.0; 2+2];
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
}
