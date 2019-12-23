use hir::{db::HirDatabase, HirDisplay};
use ra_syntax::{
    ast::{self, AstNode, LetStmt, NameOwner},
    T,
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
pub(crate) fn add_explicit_type(ctx: AssistCtx<impl HirDatabase>) -> Option<Assist> {
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
    // Assist not applicable if the type has already been specified
    if stmt.syntax().children_with_tokens().any(|child| child.kind() == T![:]) {
        return None;
    }
    // Infer type
    let db = ctx.db;
    let analyzer = ctx.source_analyzer(stmt.syntax(), None);
    let ty = analyzer.type_of(db, &expr)?;
    // Assist not applicable if the type is unknown
    if ty.contains_unknown() {
        return None;
    }

    ctx.add_assist(AssistId("add_explicit_type"), "add explicit type", |edit| {
        edit.target(pat_range);
        edit.insert(name_range.end(), format!(": {}", ty.display(db)));
    })
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
}
