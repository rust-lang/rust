//! Term search assist
use hir::term_search::{TermSearchConfig, TermSearchCtx};
use ide_db::{
    assists::{AssistId, GroupLabel},
    famous_defs::FamousDefs,
};

use itertools::Itertools;
use syntax::{AstNode, ast};

use crate::assist_context::{AssistContext, Assists};

pub(crate) fn term_search(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    let unexpanded = ctx.find_node_at_offset::<ast::MacroCall>()?;
    let syntax = unexpanded.syntax();
    let goal_range = syntax.text_range();

    let parent = syntax.parent()?;
    let scope = ctx.sema.scope(&parent)?;

    let macro_call = ctx.sema.resolve_macro_call(&unexpanded)?;

    let famous_defs = FamousDefs(&ctx.sema, scope.krate());
    let std_todo = famous_defs.core_macros_todo()?;
    let std_unimplemented = famous_defs.core_macros_unimplemented()?;

    if macro_call != std_todo && macro_call != std_unimplemented {
        return None;
    }

    let target_ty = ctx.sema.type_of_expr(&ast::Expr::cast(parent.clone())?)?.adjusted();

    let term_search_ctx = TermSearchCtx {
        sema: &ctx.sema,
        scope: &scope,
        goal: target_ty,
        config: TermSearchConfig {
            fuel: ctx.config.term_search_fuel,
            enable_borrowcheck: ctx.config.term_search_borrowck,
            ..Default::default()
        },
    };
    let paths = hir::term_search::term_search(&term_search_ctx);

    if paths.is_empty() {
        return None;
    }

    let mut formatter = |_: &hir::Type<'_>| String::from("todo!()");

    let edition = scope.krate().edition(ctx.db());
    let paths = paths
        .into_iter()
        .filter_map(|path| {
            path.gen_source_code(
                &scope,
                &mut formatter,
                ctx.config.import_path_config(),
                scope.krate().to_display_target(ctx.db()),
            )
            .ok()
        })
        .unique();

    let macro_name = macro_call.name(ctx.sema.db);
    let macro_name = macro_name.display(ctx.sema.db, edition);

    for code in paths {
        acc.add_group(
            &GroupLabel(String::from("Term search")),
            AssistId::generate("term_search"),
            format!("Replace {macro_name}!() with {code}"),
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
            r#"//- minicore: todo, unimplemented
fn f() { let a: u128 = 1; let b: u128 = todo$0!() }"#,
            r#"fn f() { let a: u128 = 1; let b: u128 = a }"#,
        )
    }

    #[test]
    fn test_complete_todo_with_msg() {
        check_assist(
            term_search,
            r#"//- minicore: todo, unimplemented
fn f() { let a: u128 = 1; let b: u128 = todo$0!("asd") }"#,
            r#"fn f() { let a: u128 = 1; let b: u128 = a }"#,
        )
    }

    #[test]
    fn test_complete_unimplemented_with_msg() {
        check_assist(
            term_search,
            r#"//- minicore: todo, unimplemented
fn f() { let a: u128 = 1; let b: u128 = unimplemented$0!("asd") }"#,
            r#"fn f() { let a: u128 = 1; let b: u128 = a }"#,
        )
    }

    #[test]
    fn test_complete_unimplemented() {
        check_assist(
            term_search,
            r#"//- minicore: todo, unimplemented
fn f() { let a: u128 = 1; let b: u128 = unimplemented$0!() }"#,
            r#"fn f() { let a: u128 = 1; let b: u128 = a }"#,
        )
    }

    #[test]
    fn test_complete_struct_field() {
        check_assist(
            term_search,
            r#"//- minicore: todo, unimplemented
struct A { pub x: i32, y: bool }
fn f() { let a = A { x: 1, y: true }; let b: i32 = todo$0!(); }"#,
            r#"struct A { pub x: i32, y: bool }
fn f() { let a = A { x: 1, y: true }; let b: i32 = a.x; }"#,
        )
    }

    #[test]
    fn test_enum_with_generics() {
        check_assist(
            term_search,
            r#"//- minicore: todo, unimplemented, option
fn f() { let a: i32 = 1; let b: Option<i32> = todo$0!(); }"#,
            r#"fn f() { let a: i32 = 1; let b: Option<i32> = None; }"#,
        )
    }

    #[test]
    fn test_enum_with_generics2() {
        check_assist(
            term_search,
            r#"//- minicore: todo, unimplemented
enum Option<T> { None, Some(T) }
fn f() { let a: i32 = 1; let b: Option<i32> = todo$0!(); }"#,
            r#"enum Option<T> { None, Some(T) }
fn f() { let a: i32 = 1; let b: Option<i32> = Option::None; }"#,
        )
    }

    #[test]
    fn test_enum_with_generics3() {
        check_assist(
            term_search,
            r#"//- minicore: todo, unimplemented
enum Option<T> { None, Some(T) }
fn f() { let a: Option<i32> = Option::None; let b: Option<Option<i32>> = todo$0!(); }"#,
            r#"enum Option<T> { None, Some(T) }
fn f() { let a: Option<i32> = Option::None; let b: Option<Option<i32>> = Option::None; }"#,
        )
    }

    #[test]
    fn test_enum_with_generics4() {
        check_assist(
            term_search,
            r#"//- minicore: todo, unimplemented
enum Foo<T = i32> { Foo(T) }
fn f() { let a = 0; let b: Foo = todo$0!(); }"#,
            r#"enum Foo<T = i32> { Foo(T) }
fn f() { let a = 0; let b: Foo = Foo::Foo(a); }"#,
        );

        check_assist(
            term_search,
            r#"//- minicore: todo, unimplemented
enum Foo<T = i32> { Foo(T) }
fn f() { let a: Foo<u32> = Foo::Foo(0); let b: Foo<u32> = todo$0!(); }"#,
            r#"enum Foo<T = i32> { Foo(T) }
fn f() { let a: Foo<u32> = Foo::Foo(0); let b: Foo<u32> = a; }"#,
        )
    }

    #[test]
    fn test_newtype() {
        check_assist(
            term_search,
            r#"//- minicore: todo, unimplemented
struct Foo(i32);
fn f() { let a: i32 = 1; let b: Foo = todo$0!(); }"#,
            r#"struct Foo(i32);
fn f() { let a: i32 = 1; let b: Foo = Foo(a); }"#,
        )
    }

    #[test]
    fn test_shadowing() {
        check_assist(
            term_search,
            r#"//- minicore: todo, unimplemented
fn f() { let a: i32 = 1; let b: i32 = 2; let a: u32 = 0; let c: i32 = todo$0!(); }"#,
            r#"fn f() { let a: i32 = 1; let b: i32 = 2; let a: u32 = 0; let c: i32 = b; }"#,
        )
    }

    #[test]
    fn test_famous_bool() {
        check_assist(
            term_search,
            r#"//- minicore: todo, unimplemented
fn f() { let a: bool = todo$0!(); }"#,
            r#"fn f() { let a: bool = true; }"#,
        )
    }

    #[test]
    fn test_fn_with_reference_types() {
        check_assist(
            term_search,
            r#"//- minicore: todo, unimplemented
fn f(a: &i32) -> f32 { a as f32 }
fn g() { let a = 1; let b: f32 = todo$0!(); }"#,
            r#"fn f(a: &i32) -> f32 { a as f32 }
fn g() { let a = 1; let b: f32 = f(&a); }"#,
        )
    }

    #[test]
    fn test_fn_with_reference_types2() {
        check_assist(
            term_search,
            r#"//- minicore: todo, unimplemented
fn f(a: &i32) -> f32 { a as f32 }
fn g() { let a = &1; let b: f32 = todo$0!(); }"#,
            r#"fn f(a: &i32) -> f32 { a as f32 }
fn g() { let a = &1; let b: f32 = f(a); }"#,
        )
    }

    #[test]
    fn test_fn_with_reference_types3() {
        check_assist_not_applicable(
            term_search,
            r#"//- minicore: todo, unimplemented
            fn f(a: &i32) -> f32 { a as f32 }
            fn g() { let a = &mut 1; let b: f32 = todo$0!(); }"#,
        )
    }

    #[test]
    fn test_tuple_simple() {
        check_assist(
            term_search,
            r#"//- minicore: todo, unimplemented
fn f() { let a = 1; let b = 0.0; let c: (i32, f64) = todo$0!(); }"#,
            r#"fn f() { let a = 1; let b = 0.0; let c: (i32, f64) = (a, b); }"#,
        )
    }

    #[test]
    fn test_tuple_nested() {
        check_assist(
            term_search,
            r#"//- minicore: todo, unimplemented
fn f() { let a = 1; let b = 0.0; let c: (i32, (i32, f64)) = todo$0!(); }"#,
            r#"fn f() { let a = 1; let b = 0.0; let c: (i32, (i32, f64)) = (a, (a, b)); }"#,
        )
    }

    #[test]
    fn test_tuple_struct_with_generics() {
        check_assist(
            term_search,
            r#"//- minicore: todo, unimplemented
struct Foo<T>(T);
fn f() { let a = 1; let b: Foo<i32> = todo$0!(); }"#,
            r#"struct Foo<T>(T);
fn f() { let a = 1; let b: Foo<i32> = Foo(a); }"#,
        )
    }

    #[test]
    fn test_struct_assoc_item() {
        check_assist(
            term_search,
            r#"//- minicore: todo, unimplemented
struct Foo;
impl Foo { const FOO: i32 = 0; }
fn f() { let a: i32 = todo$0!(); }"#,
            r#"struct Foo;
impl Foo { const FOO: i32 = 0; }
fn f() { let a: i32 = Foo::FOO; }"#,
        )
    }

    #[test]
    fn test_trait_assoc_item() {
        check_assist(
            term_search,
            r#"//- minicore: todo, unimplemented
struct Foo;
trait Bar { const BAR: i32; }
impl Bar for Foo { const BAR: i32 = 0; }
fn f() { let a: i32 = todo$0!(); }"#,
            r#"struct Foo;
trait Bar { const BAR: i32; }
impl Bar for Foo { const BAR: i32 = 0; }
fn f() { let a: i32 = Foo::BAR; }"#,
        )
    }
}
