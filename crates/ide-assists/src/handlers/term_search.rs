//! Term search assist
use hir::term_search::TermSearchCtx;
use ide_db::assists::{AssistId, AssistKind, GroupLabel};

use itertools::Itertools;
use syntax::{ast, AstNode};

use crate::assist_context::{AssistContext, Assists};

pub(crate) fn term_search(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    let unexpanded = ctx.find_node_at_offset::<ast::MacroCall>()?;
    let syntax = unexpanded.syntax();
    let goal_range = syntax.text_range();

    let excl = unexpanded.excl_token()?;
    let macro_name_token = excl.prev_token()?;
    let name = macro_name_token.text();
    if name != "todo" {
        return None;
    }

    let parent = syntax.parent()?;
    let target_ty = ctx.sema.type_of_expr(&ast::Expr::cast(parent.clone())?)?.adjusted();

    let scope = ctx.sema.scope(&parent)?;

    let term_search_ctx = TermSearchCtx {
        sema: &ctx.sema,
        scope: &scope,
        goal: target_ty,
        config: Default::default(),
    };
    let paths = hir::term_search::term_search(term_search_ctx);

    if paths.is_empty() {
        return None;
    }

    let mut formatter = |_: &hir::Type| String::from("todo!()");
    for path in paths.iter().unique() {
        let code = path.gen_source_code(&scope, &mut formatter);
        acc.add_group(
            &GroupLabel(String::from("Term search")),
            AssistId("term_search", AssistKind::Generate),
            format!("Replace todo!() with {code}"),
            goal_range,
            |builder| {
                builder.replace(goal_range, code);
            },
        );
    }

    Some(())
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_assist, check_assist_not_applicable};

    use super::*;

    #[test]
    fn test_complete_local() {
        check_assist(
            term_search,
            "macro_rules! todo { () => (_) }; fn f() { let a: u128 = 1; let b: u128 = todo$0!() }",
            "macro_rules! todo { () => (_) }; fn f() { let a: u128 = 1; let b: u128 = a }",
        )
    }

    #[test]
    fn test_complete_todo_with_msg() {
        check_assist(
            term_search,
            "macro_rules! todo { ($($arg:tt)+) => (_) }; fn f() { let a: u128 = 1; let b: u128 = todo$0!(\"asd\") }",
            "macro_rules! todo { ($($arg:tt)+) => (_) }; fn f() { let a: u128 = 1; let b: u128 = a }",
        )
    }

    #[test]
    fn test_complete_struct_field() {
        check_assist(
            term_search,
            r#"macro_rules! todo { () => (_) };
            struct A { pub x: i32, y: bool }
            fn f() { let a = A { x: 1, y: true }; let b: i32 = todo$0!(); }"#,
            r#"macro_rules! todo { () => (_) };
            struct A { pub x: i32, y: bool }
            fn f() { let a = A { x: 1, y: true }; let b: i32 = a.x; }"#,
        )
    }

    #[test]
    fn test_enum_with_generics() {
        check_assist(
            term_search,
            r#"macro_rules! todo { () => (_) };
            enum Option<T> { Some(T), None }
            fn f() { let a: i32 = 1; let b: Option<i32> = todo$0!(); }"#,
            r#"macro_rules! todo { () => (_) };
            enum Option<T> { Some(T), None }
            fn f() { let a: i32 = 1; let b: Option<i32> = Option::None; }"#,
        )
    }

    #[test]
    fn test_enum_with_generics2() {
        check_assist(
            term_search,
            r#"macro_rules! todo { () => (_) };
            enum Option<T> { None, Some(T) }
            fn f() { let a: i32 = 1; let b: Option<i32> = todo$0!(); }"#,
            r#"macro_rules! todo { () => (_) };
            enum Option<T> { None, Some(T) }
            fn f() { let a: i32 = 1; let b: Option<i32> = Option::Some(a); }"#,
        )
    }

    #[test]
    fn test_enum_with_generics3() {
        check_assist(
            term_search,
            r#"macro_rules! todo { () => (_) };
            enum Option<T> { None, Some(T) }
            fn f() { let a: Option<i32> = Option::None; let b: Option<Option<i32>> = todo$0!(); }"#,
            r#"macro_rules! todo { () => (_) };
            enum Option<T> { None, Some(T) }
            fn f() { let a: Option<i32> = Option::None; let b: Option<Option<i32>> = Option::Some(a); }"#,
        )
    }

    #[test]
    fn test_enum_with_generics4() {
        check_assist(
            term_search,
            r#"macro_rules! todo { () => (_) };
            enum Foo<T = i32> { Foo(T) }
            fn f() { let a = 0; let b: Foo = todo$0!(); }"#,
            r#"macro_rules! todo { () => (_) };
            enum Foo<T = i32> { Foo(T) }
            fn f() { let a = 0; let b: Foo = Foo::Foo(a); }"#,
        );

        check_assist(
            term_search,
            r#"macro_rules! todo { () => (_) };
            enum Foo<T = i32> { Foo(T) }
            fn f() { let a: Foo<u32> = Foo::Foo(0); let b: Foo<u32> = todo$0!(); }"#,
            r#"macro_rules! todo { () => (_) };
            enum Foo<T = i32> { Foo(T) }
            fn f() { let a: Foo<u32> = Foo::Foo(0); let b: Foo<u32> = a; }"#,
        )
    }

    #[test]
    fn test_newtype() {
        check_assist(
            term_search,
            r#"macro_rules! todo { () => (_) };
            struct Foo(i32);
            fn f() { let a: i32 = 1; let b: Foo = todo$0!(); }"#,
            r#"macro_rules! todo { () => (_) };
            struct Foo(i32);
            fn f() { let a: i32 = 1; let b: Foo = Foo(a); }"#,
        )
    }

    #[test]
    fn test_shadowing() {
        check_assist(
            term_search,
            r#"macro_rules! todo { () => (_) };
            fn f() { let a: i32 = 1; let b: i32 = 2; let a: u32 = 0; let c: i32 = todo$0!(); }"#,
            r#"macro_rules! todo { () => (_) };
            fn f() { let a: i32 = 1; let b: i32 = 2; let a: u32 = 0; let c: i32 = b; }"#,
        )
    }

    #[test]
    fn test_famous_bool() {
        check_assist(
            term_search,
            r#"macro_rules! todo { () => (_) };
            fn f() { let a: bool = todo$0!(); }"#,
            r#"macro_rules! todo { () => (_) };
            fn f() { let a: bool = false; }"#,
        )
    }

    #[test]
    fn test_fn_with_reference_types() {
        check_assist(
            term_search,
            r#"macro_rules! todo { () => (_) };
            fn f(a: &i32) -> f32 { a as f32 }
            fn g() { let a = 1; let b: f32 = todo$0!(); }"#,
            r#"macro_rules! todo { () => (_) };
            fn f(a: &i32) -> f32 { a as f32 }
            fn g() { let a = 1; let b: f32 = f(&a); }"#,
        )
    }

    #[test]
    fn test_fn_with_reference_types2() {
        check_assist(
            term_search,
            r#"macro_rules! todo { () => (_) };
            fn f(a: &i32) -> f32 { a as f32 }
            fn g() { let a = &1; let b: f32 = todo$0!(); }"#,
            r#"macro_rules! todo { () => (_) };
            fn f(a: &i32) -> f32 { a as f32 }
            fn g() { let a = &1; let b: f32 = f(a); }"#,
        )
    }

    #[test]
    fn test_fn_with_reference_types3() {
        check_assist_not_applicable(
            term_search,
            r#"macro_rules! todo { () => (_) };
            fn f(a: &i32) -> f32 { a as f32 }
            fn g() { let a = &mut 1; let b: f32 = todo$0!(); }"#,
        )
    }
}
