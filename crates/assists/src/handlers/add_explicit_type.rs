use hir::HirDisplay;
use syntax::{
    ast::{self, AstNode, LetStmt, NameOwner},
    TextRange,
};

use crate::{AssistContext, AssistId, AssistKind, Assists};

// Assist: add_explicit_type
//
// Specify type for a let binding.
//
// ```
// fn main() {
//     let x<|> = 92;
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
    // The binding must have a name
    let name = pat.name()?;
    let name_range = name.syntax().text_range();
    let stmt_range = let_stmt.syntax().text_range();
    let eq_range = let_stmt.eq_token()?.text_range();
    // Assist should only be applicable if cursor is between 'let' and '='
    let let_range = TextRange::new(stmt_range.start(), eq_range.start());
    let cursor_in_range = let_range.contains_range(ctx.frange.range);
    if !cursor_in_range {
        return None;
    }
    // Assist not applicable if the type has already been specified
    // and it has no placeholders
    let ascribed_ty = let_stmt.ty();
    if let Some(ty) = &ascribed_ty {
        if ty.syntax().descendants().find_map(ast::InferType::cast).is_none() {
            return None;
        }
    }
    // Infer type
    let ty = ctx.sema.type_of_expr(&expr)?;

    if ty.contains_unknown() || ty.is_closure() {
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
                builder.insert(name_range.end(), format!(": {}", inferred_type));
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
        check_assist_target(add_explicit_type, "fn f() { let a<|> = 1; }", "a");
    }

    #[test]
    fn add_explicit_type_works_for_simple_expr() {
        check_assist(add_explicit_type, "fn f() { let a<|> = 1; }", "fn f() { let a: i32 = 1; }");
    }

    #[test]
    fn add_explicit_type_works_for_underscore() {
        check_assist(
            add_explicit_type,
            "fn f() { let a<|>: _ = 1; }",
            "fn f() { let a: i32 = 1; }",
        );
    }

    #[test]
    fn add_explicit_type_works_for_nested_underscore() {
        check_assist(
            add_explicit_type,
            r#"
            enum Option<T> {
                Some(T),
                None
            }

            fn f() {
                let a<|>: Option<_> = Option::Some(1);
            }"#,
            r#"
            enum Option<T> {
                Some(T),
                None
            }

            fn f() {
                let a: Option<i32> = Option::Some(1);
            }"#,
        );
    }

    #[test]
    fn add_explicit_type_works_for_macro_call() {
        check_assist(
            add_explicit_type,
            r"macro_rules! v { () => {0u64} } fn f() { let a<|> = v!(); }",
            r"macro_rules! v { () => {0u64} } fn f() { let a: u64 = v!(); }",
        );
    }

    #[test]
    fn add_explicit_type_works_for_macro_call_recursive() {
        check_assist(
            add_explicit_type,
            r#"macro_rules! u { () => {0u64} } macro_rules! v { () => {u!()} } fn f() { let a<|> = v!(); }"#,
            r#"macro_rules! u { () => {0u64} } macro_rules! v { () => {u!()} } fn f() { let a: u64 = v!(); }"#,
        );
    }

    #[test]
    fn add_explicit_type_not_applicable_if_ty_not_inferred() {
        check_assist_not_applicable(add_explicit_type, "fn f() { let a<|> = None; }");
    }

    #[test]
    fn add_explicit_type_not_applicable_if_ty_already_specified() {
        check_assist_not_applicable(add_explicit_type, "fn f() { let a<|>: i32 = 1; }");
    }

    #[test]
    fn add_explicit_type_not_applicable_if_specified_ty_is_tuple() {
        check_assist_not_applicable(add_explicit_type, "fn f() { let a<|>: (i32, i32) = (3, 4); }");
    }

    #[test]
    fn add_explicit_type_not_applicable_if_cursor_after_equals() {
        check_assist_not_applicable(
            add_explicit_type,
            "fn f() {let a =<|> match 1 {2 => 3, 3 => 5};}",
        )
    }

    #[test]
    fn add_explicit_type_not_applicable_if_cursor_before_let() {
        check_assist_not_applicable(
            add_explicit_type,
            "fn f() <|>{let a = match 1 {2 => 3, 3 => 5};}",
        )
    }

    #[test]
    fn closure_parameters_are_not_added() {
        check_assist_not_applicable(
            add_explicit_type,
            r#"
fn main() {
    let multiply_by_two<|> = |i| i * 3;
    let six = multiply_by_two(2);
}"#,
        )
    }

    #[test]
    fn default_generics_should_not_be_added() {
        check_assist(
            add_explicit_type,
            r#"
struct Test<K, T = u8> {
    k: K,
    t: T,
}

fn main() {
    let test<|> = Test { t: 23u8, k: 33 };
}"#,
            r#"
struct Test<K, T = u8> {
    k: K,
    t: T,
}

fn main() {
    let test: Test<i32> = Test { t: 23u8, k: 33 };
}"#,
        );
    }
}
