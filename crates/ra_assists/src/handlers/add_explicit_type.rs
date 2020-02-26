use hir::HirDisplay;
use ra_syntax::{
    ast::{self, AstNode, LetStmt, NameOwner, TypeAscriptionOwner},
    TextRange,
};

use crate::{Assist, AssistCtx, AssistId};

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
pub(crate) fn add_explicit_type(ctx: AssistCtx) -> Option<Assist> {
    let stmt = ctx.find_node_at_offset::<LetStmt>()?;
    let expr = stmt.initializer()?;
    let pat = stmt.pat()?;
    // Must be a binding
    let pat = match pat {
        ast::Pat::BindPat(bind_pat) => bind_pat,
        _ => return None,
    };
    let pat_range = pat.syntax().text_range();
    // The binding must have a name
    let name = pat.name()?;
    let name_range = name.syntax().text_range();
    let stmt_range = stmt.syntax().text_range();
    let eq_range = stmt.eq_token()?.text_range();
    // Assist should only be applicable if cursor is between 'let' and '='
    let let_range = TextRange::from_to(stmt_range.start(), eq_range.start());
    let cursor_in_range = ctx.frange.range.is_subrange(&let_range);
    if !cursor_in_range {
        return None;
    }
    // Assist not applicable if the type has already been specified
    // and it has no placeholders
    let ascribed_ty = stmt.ascribed_type();
    if let Some(ref ty) = ascribed_ty {
        if ty.syntax().descendants().find_map(ast::PlaceholderType::cast).is_none() {
            return None;
        }
    }
    // Infer type
    let ty = ctx.sema.type_of_expr(&expr)?;
    // Assist not applicable if the type is unknown
    if ty.contains_unknown() {
        return None;
    }

    let db = ctx.db;
    ctx.add_assist(
        AssistId("add_explicit_type"),
        format!("Insert explicit type '{}'", ty.display(db)),
        |edit| {
            edit.target(pat_range);
            if let Some(ascribed_ty) = ascribed_ty {
                edit.replace(ascribed_ty.syntax().text_range(), format!("{}", ty.display(db)));
            } else {
                edit.insert(name_range.end(), format!(": {}", ty.display(db)));
            }
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::helpers::{check_assist, check_assist_not_applicable, check_assist_target};

    #[test]
    fn add_explicit_type_target() {
        check_assist_target(add_explicit_type, "fn f() { let a<|> = 1; }", "a");
    }

    #[test]
    fn add_explicit_type_works_for_simple_expr() {
        check_assist(
            add_explicit_type,
            "fn f() { let a<|> = 1; }",
            "fn f() { let a<|>: i32 = 1; }",
        );
    }

    #[test]
    fn add_explicit_type_works_for_underscore() {
        check_assist(
            add_explicit_type,
            "fn f() { let a<|>: _ = 1; }",
            "fn f() { let a<|>: i32 = 1; }",
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
                let a<|>: Option<i32> = Option::Some(1);
            }"#,
        );
    }

    #[test]
    fn add_explicit_type_works_for_macro_call() {
        check_assist(
            add_explicit_type,
            "macro_rules! v { () => {0u64} } fn f() { let a<|> = v!(); }",
            "macro_rules! v { () => {0u64} } fn f() { let a<|>: u64 = v!(); }",
        );
    }

    #[test]
    fn add_explicit_type_works_for_macro_call_recursive() {
        check_assist(
            add_explicit_type,
            "macro_rules! u { () => {0u64} } macro_rules! v { () => {u!()} } fn f() { let a<|> = v!(); }",
            "macro_rules! u { () => {0u64} } macro_rules! v { () => {u!()} } fn f() { let a<|>: u64 = v!(); }",
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
}
