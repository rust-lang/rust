use hir::HasCrate;
use syntax::{AstNode, ast};

use crate::{AssistContext, AssistId, Assists};

// Assist: inline_const_as_literal
//
// Evaluate and inline const variable as literal.
//
// ```
// const STRING: &str = "Hello, World!";
//
// fn something() -> &'static str {
//     STRING$0
// }
// ```
// ->
// ```
// const STRING: &str = "Hello, World!";
//
// fn something() -> &'static str {
//     "Hello, World!"
// }
// ```
pub(crate) fn inline_const_as_literal(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    let variable = ctx.find_node_at_offset::<ast::PathExpr>()?;

    if let hir::PathResolution::Def(hir::ModuleDef::Const(konst)) =
        ctx.sema.resolve_path(&variable.path()?)?
    {
        let konst_ty = konst.ty(ctx.sema.db);

        // Used as the upper limit for recursive calls if no TCO is available
        let fuel = 20;

        // There is no way to have a const static reference to a type that contains a interior
        // mutability cell.

        // FIXME: Add support to handle type aliases for builtin scalar types.
        validate_type_recursively(ctx, Some(&konst_ty), false, fuel)?;

        let value = konst
            .eval(ctx.sema.db)
            .ok()?
            .render(ctx.sema.db, konst.krate(ctx.sema.db).to_display_target(ctx.sema.db));

        let id = AssistId::refactor_inline("inline_const_as_literal");

        let label = "Inline const as literal".to_owned();
        let target = variable.syntax().text_range();

        return acc.add(id, label, target, |edit| {
            edit.replace(variable.syntax().text_range(), value);
        });
    }
    None
}

fn validate_type_recursively(
    ctx: &AssistContext<'_>,
    ty_hir: Option<&hir::Type<'_>>,
    refed: bool,
    fuel: i32,
) -> Option<()> {
    match (fuel > 0, ty_hir) {
        (true, Some(ty)) if ty.is_reference() => validate_type_recursively(
            ctx,
            ty.as_reference().map(|(ty, _)| ty).as_ref(),
            true,
            // FIXME: Saving fuel when `&` repeating might not be a good idea if there's no TCO.
            if refed { fuel } else { fuel - 1 },
        ),
        (true, Some(ty)) if ty.is_array() => validate_type_recursively(
            ctx,
            ty.as_array(ctx.db()).map(|(ty, _)| ty).as_ref(),
            false,
            fuel - 1,
        ),
        (true, Some(ty)) if ty.is_tuple() => ty
            .tuple_fields(ctx.db())
            .iter()
            .all(|ty| validate_type_recursively(ctx, Some(ty), false, fuel - 1).is_some())
            .then_some(()),
        (true, Some(ty)) if refed && ty.is_slice() => {
            validate_type_recursively(ctx, ty.as_slice().as_ref(), false, fuel - 1)
        }
        (_, Some(ty)) => match ty.as_builtin() {
            // `const A: str` is not correct, but `const A: &builtin` is.
            Some(builtin) if refed || !builtin.is_str() => Some(()),
            _ => None,
        },
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::{check_assist, check_assist_not_applicable};

    const NUMBER: u8 = 1;
    const BOOL: u8 = 2;
    const STR: u8 = 4;
    const CHAR: u8 = 8;

    const TEST_PAIRS: &[(&str, &str, u8)] = &[
        ("u8", "0", NUMBER),
        ("u16", "0", NUMBER),
        ("u32", "0", NUMBER),
        ("u64", "0", NUMBER),
        ("u128", "0", NUMBER),
        ("usize", "0", NUMBER),
        ("usize", "16", NUMBER),
        ("i8", "0", NUMBER),
        ("i16", "0", NUMBER),
        ("i32", "0", NUMBER),
        ("i64", "0", NUMBER),
        ("i128", "0", NUMBER),
        ("isize", "0", NUMBER),
        ("isize", "16", NUMBER),
        ("bool", "false", BOOL),
        ("&str", "\"str\"", STR),
        ("char", "'c'", CHAR),
    ];

    // -----------Not supported-----------
    #[test]
    fn inline_const_as_literal_const_fn_call_slice() {
        TEST_PAIRS.iter().for_each(|(ty, val, _)| {
            check_assist_not_applicable(
                inline_const_as_literal,
                &format!(
                    r#"
                    const fn abc() -> &[{ty}] {{ &[{val}] }}
                    const ABC: &[{ty}] = abc();
                    fn a() {{ A$0BC }}
                    "#
                ),
            );
        });
    }

    #[test]
    fn inline_const_as_literal_expr_as_str_lit_not_applicable_const() {
        check_assist_not_applicable(
            inline_const_as_literal,
            r#"
            const STR$0ING: &str = "Hello, World!";

            fn something() -> &'static str {
                STRING
            }
            "#,
        );
    }

    #[test]
    fn inline_const_as_struct_() {
        check_assist_not_applicable(
            inline_const_as_literal,
            r#"
            struct A;
            const STRUKT: A = A;

            fn something() -> A {
                STRU$0KT
            }
            "#,
        );
    }

    #[test]
    fn inline_const_as_enum_() {
        check_assist_not_applicable(
            inline_const_as_literal,
            r#"
            enum A { A, B, C }
            const ENUM: A = A::A;

            fn something() -> A {
                EN$0UM
            }
            "#,
        );
    }

    #[test]
    fn inline_const_as_tuple_closure() {
        check_assist_not_applicable(
            inline_const_as_literal,
            r#"
            const CLOSURE: (&dyn Fn(i32) -> i32) = (&|num| -> i32 { num });
            fn something() -> (&dyn Fn(i32) -> i32) {
                STRU$0KT
            }
            "#,
        );
    }

    #[test]
    fn inline_const_as_closure_() {
        check_assist_not_applicable(
            inline_const_as_literal,
            r#"
            const CLOSURE: &dyn Fn(i32) -> i32 = &|num| -> i32 { num };
            fn something() -> &dyn Fn(i32) -> i32 {
                STRU$0KT
            }
            "#,
        );
    }

    #[test]
    fn inline_const_as_fn_() {
        check_assist_not_applicable(
            inline_const_as_literal,
            r#"
            struct S(i32);
            const CON: fn(i32) -> S = S;
            fn something() {
                let x = CO$0N;
            }
            "#,
        );
    }

    // ----------------------------

    #[test]
    fn inline_const_as_literal_const_expr() {
        TEST_PAIRS.iter().for_each(|(ty, val, _)| {
            check_assist(
                inline_const_as_literal,
                &format!(
                    r#"
                    const ABC: {ty} = {val};
                    fn a() {{ A$0BC }}
                    "#
                ),
                &format!(
                    r#"
                    const ABC: {ty} = {val};
                    fn a() {{ {val} }}
                    "#
                ),
            );
        });
    }

    #[test]
    fn inline_const_as_literal_const_block_expr() {
        TEST_PAIRS.iter().for_each(|(ty, val, _)| {
            check_assist(
                inline_const_as_literal,
                &format!(
                    r#"
                    const ABC: {ty} = {{ {val} }};
                    fn a() {{ A$0BC }}
                    "#
                ),
                &format!(
                    r#"
                    const ABC: {ty} = {{ {val} }};
                    fn a() {{ {val} }}
                    "#
                ),
            );
        });
    }

    #[test]
    fn inline_const_as_literal_const_block_eval_expr() {
        TEST_PAIRS.iter().for_each(|(ty, val, _)| {
            check_assist(
                inline_const_as_literal,
                &format!(
                    r#"
                    const ABC: {ty} = {{ true; {val} }};
                    fn a() {{ A$0BC }}
                    "#
                ),
                &format!(
                    r#"
                    const ABC: {ty} = {{ true; {val} }};
                    fn a() {{ {val} }}
                    "#
                ),
            );
        });
    }

    #[test]
    fn inline_const_as_literal_const_block_eval_block_expr() {
        TEST_PAIRS.iter().for_each(|(ty, val, _)| {
            check_assist(
                inline_const_as_literal,
                &format!(
                    r#"
                    const ABC: {ty} = {{ true; {{ {val} }} }};
                    fn a() {{ A$0BC }}
                    "#
                ),
                &format!(
                    r#"
                    const ABC: {ty} = {{ true; {{ {val} }} }};
                    fn a() {{ {val} }}
                    "#
                ),
            );
        });
    }

    #[test]
    fn inline_const_as_literal_const_fn_call_block_nested_builtin() {
        TEST_PAIRS.iter().for_each(|(ty, val, _)| {
            check_assist(
                inline_const_as_literal,
                &format!(
                    r#"
                    const fn abc() -> {ty} {{ {{ {{ {{ {val} }} }} }} }}
                    const ABC: {ty} = abc();
                    fn a() {{ A$0BC }}
                    "#
                ),
                &format!(
                    r#"
                    const fn abc() -> {ty} {{ {{ {{ {{ {val} }} }} }} }}
                    const ABC: {ty} = abc();
                    fn a() {{ {val} }}
                    "#
                ),
            );
        });
    }

    #[test]
    fn inline_const_as_literal_const_fn_call_tuple() {
        TEST_PAIRS.iter().for_each(|(ty, val, _)| {
            check_assist(
                inline_const_as_literal,
                &format!(
                    r#"
                    const fn abc() -> ({ty}, {ty}) {{ ({val}, {val}) }}
                    const ABC: ({ty}, {ty}) = abc();
                    fn a() {{ A$0BC }}
                    "#
                ),
                &format!(
                    r#"
                    const fn abc() -> ({ty}, {ty}) {{ ({val}, {val}) }}
                    const ABC: ({ty}, {ty}) = abc();
                    fn a() {{ ({val}, {val}) }}
                    "#
                ),
            );
        });
    }

    #[test]
    fn inline_const_as_literal_const_fn_call_builtin() {
        TEST_PAIRS.iter().for_each(|(ty, val, _)| {
            check_assist(
                inline_const_as_literal,
                &format!(
                    r#"
					const fn abc() -> {ty} {{ {val} }}
					const ABC: {ty} = abc();
					fn a() {{ A$0BC }}
					"#
                ),
                &format!(
                    r#"
					const fn abc() -> {ty} {{ {val} }}
					const ABC: {ty} = abc();
					fn a() {{ {val} }}
					"#
                ),
            );
        });
    }

    #[test]
    fn inline_const_as_literal_scalar_operators() {
        check_assist(
            inline_const_as_literal,
            r#"
            const ABC: i32 = 1 + 2 + 3;
            fn a() { A$0BC }
            "#,
            r#"
            const ABC: i32 = 1 + 2 + 3;
            fn a() { 6 }
            "#,
        );
    }
    #[test]
    fn inline_const_as_literal_block_scalar_calculate_expr() {
        check_assist(
            inline_const_as_literal,
            r#"
            const ABC: i32 = { 1 + 2 + 3 };
            fn a() { A$0BC }
            "#,
            r#"
            const ABC: i32 = { 1 + 2 + 3 };
            fn a() { 6 }
            "#,
        );
    }

    #[test]
    fn inline_const_as_literal_block_scalar_calculate_param_expr() {
        check_assist(
            inline_const_as_literal,
            r#"
            const ABC: i32 = { (1 + 2 + 3) };
            fn a() { A$0BC }
            "#,
            r#"
            const ABC: i32 = { (1 + 2 + 3) };
            fn a() { 6 }
            "#,
        );
    }

    #[test]
    fn inline_const_as_literal_block_tuple_scalar_calculate_block_expr() {
        check_assist(
            inline_const_as_literal,
            r#"
            const ABC: (i32, i32) = { (1, { 2 + 3 }) };
            fn a() { A$0BC }
            "#,
            r#"
            const ABC: (i32, i32) = { (1, { 2 + 3 }) };
            fn a() { (1, 5) }
            "#,
        );
    }

    // FIXME: Add support for nested ref slices when using `render_eval`
    #[test]
    fn inline_const_as_literal_block_slice() {
        check_assist_not_applicable(
            inline_const_as_literal,
            r#"
            const ABC: &[&[&[&[&[&[i32]]]]]] = { &[&[&[&[&[&[10, 20, 30]]]]]] };
            fn a() { A$0BC }
            "#,
        );
    }

    // FIXME: Add support for unary tuple expressions when using `render_eval`.
    // `const fn abc() -> (i32) { (1) }` will results in `1` instead of `(1)` because it's evaluated
    // as a paren expr.
    #[test]
    fn inline_const_as_literal_block_tuple() {
        check_assist(
            inline_const_as_literal,
            r#"
            const ABC: (([i32; 3]), (i32), ((&str, i32), i32), i32) = { (([1, 2, 3]), (10), (("hello", 10), 20), 30) };
            fn a() { A$0BC }
            "#,
            r#"
            const ABC: (([i32; 3]), (i32), ((&str, i32), i32), i32) = { (([1, 2, 3]), (10), (("hello", 10), 20), 30) };
            fn a() { ([1, 2, 3], 10, (("hello", 10), 20), 30) }
            "#,
        );
    }

    #[test]
    fn inline_const_as_literal_block_slice_single() {
        check_assist(
            inline_const_as_literal,
            r#"
            const ABC: [i32; 1] = { [10] };
            fn a() { A$0BC }
            "#,
            r#"
            const ABC: [i32; 1] = { [10] };
            fn a() { [10] }
            "#,
        );
    }

    #[test]
    fn inline_const_as_literal_block_array() {
        check_assist(
            inline_const_as_literal,
            r#"
            const ABC: [[[i32; 1]; 1]; 1] = { [[[10]]] };
            fn a() { A$0BC }
            "#,
            r#"
            const ABC: [[[i32; 1]; 1]; 1] = { [[[10]]] };
            fn a() { [[[10]]] }
            "#,
        );
    }

    #[test]
    fn inline_const_as_literal_block_recursive() {
        check_assist(
            inline_const_as_literal,
            r#"
            const ABC: &str = { { { { "hello" } } } };
            fn a() { A$0BC }
            "#,
            r#"
            const ABC: &str = { { { { "hello" } } } };
            fn a() { "hello" }
            "#,
        );
    }

    #[test]
    fn inline_const_as_literal_expr_as_str_lit() {
        check_assist(
            inline_const_as_literal,
            r#"
            const STRING: &str = "Hello, World!";

            fn something() -> &'static str {
                STR$0ING
            }
            "#,
            r#"
            const STRING: &str = "Hello, World!";

            fn something() -> &'static str {
                "Hello, World!"
            }
            "#,
        );
    }

    #[test]
    fn inline_const_as_literal_eval_const_block_expr_to_str_lit() {
        check_assist(
            inline_const_as_literal,
            r#"
            const STRING: &str = {
                let x = 9;
                if x + 10 == 21 {
                    "Hello, World!"
                } else {
                    "World, Hello!"
                }
            };

            fn something() -> &'static str {
                STR$0ING
            }
            "#,
            r#"
            const STRING: &str = {
                let x = 9;
                if x + 10 == 21 {
                    "Hello, World!"
                } else {
                    "World, Hello!"
                }
            };

            fn something() -> &'static str {
                "World, Hello!"
            }
            "#,
        );
    }

    #[test]
    fn inline_const_as_literal_eval_const_block_macro_expr_to_str_lit() {
        check_assist(
            inline_const_as_literal,
            r#"
            macro_rules! co {() => {"World, Hello!"};}
            const STRING: &str = { co!() };

            fn something() -> &'static str {
                STR$0ING
            }
            "#,
            r#"
            macro_rules! co {() => {"World, Hello!"};}
            const STRING: &str = { co!() };

            fn something() -> &'static str {
                "World, Hello!"
            }
            "#,
        );
    }

    #[test]
    fn inline_const_as_literal_eval_const_match_expr_to_str_lit() {
        check_assist(
            inline_const_as_literal,
            r#"
            const STRING: &str = match 9 + 10 {
                0..18 => "Hello, World!",
                _ => "World, Hello!"
            };

            fn something() -> &'static str {
                STR$0ING
            }
            "#,
            r#"
            const STRING: &str = match 9 + 10 {
                0..18 => "Hello, World!",
                _ => "World, Hello!"
            };

            fn something() -> &'static str {
                "World, Hello!"
            }
            "#,
        );
    }

    #[test]
    fn inline_const_as_literal_eval_const_if_expr_to_str_lit() {
        check_assist(
            inline_const_as_literal,
            r#"
            const STRING: &str = if 1 + 2 == 4 {
                "Hello, World!"
            } else {
                "World, Hello!"
            }

            fn something() -> &'static str {
                STR$0ING
            }
            "#,
            r#"
            const STRING: &str = if 1 + 2 == 4 {
                "Hello, World!"
            } else {
                "World, Hello!"
            }

            fn something() -> &'static str {
                "World, Hello!"
            }
            "#,
        );
    }

    #[test]
    fn inline_const_as_literal_eval_const_macro_expr_to_str_lit() {
        check_assist(
            inline_const_as_literal,
            r#"
            macro_rules! co {() => {"World, Hello!"};}
            const STRING: &str = co!();

            fn something() -> &'static str {
                STR$0ING
            }
            "#,
            r#"
            macro_rules! co {() => {"World, Hello!"};}
            const STRING: &str = co!();

            fn something() -> &'static str {
                "World, Hello!"
            }
            "#,
        );
    }

    #[test]
    fn inline_const_as_literal_eval_const_call_expr_to_str_lit() {
        check_assist(
            inline_const_as_literal,
            r#"
            const fn const_call() -> &'static str {"World, Hello!"}
            const STRING: &str = const_call();

            fn something() -> &'static str {
                STR$0ING
            }
            "#,
            r#"
            const fn const_call() -> &'static str {"World, Hello!"}
            const STRING: &str = const_call();

            fn something() -> &'static str {
                "World, Hello!"
            }
            "#,
        );
    }

    #[test]
    fn inline_const_as_literal_expr_as_str_lit_not_applicable() {
        check_assist_not_applicable(
            inline_const_as_literal,
            r#"
            const STRING: &str = "Hello, World!";

            fn something() -> &'static str {
                STRING $0
            }
            "#,
        );
    }
}
