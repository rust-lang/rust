use std::iter;

use hir::HasSource;
use ide_db::{
    famous_defs::FamousDefs,
    syntax_helpers::node_ext::{for_each_tail_expr, walk_expr},
};
use itertools::Itertools;
use syntax::{
    ast::{self, make, Expr, HasGenericParams},
    match_ast, ted, AstNode, ToSmolStr,
};

use crate::{AssistContext, AssistId, AssistKind, Assists};

// Assist: wrap_return_type_in_option
//
// Wrap the function's return type into Option.
//
// ```
// # //- minicore: option
// fn foo() -> i32$0 { 42i32 }
// ```
// ->
// ```
// fn foo() -> Option<i32> { Some(42i32) }
// ```
pub(crate) fn wrap_return_type_in_option(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    let ret_type = ctx.find_node_at_offset::<ast::RetType>()?;
    let parent = ret_type.syntax().parent()?;
    let body = match_ast! {
        match parent {
            ast::Fn(func) => func.body()?,
            ast::ClosureExpr(closure) => match closure.body()? {
                Expr::BlockExpr(block) => block,
                // closures require a block when a return type is specified
                _ => return None,
            },
            _ => return None,
        }
    };

    let type_ref = &ret_type.ty()?;
    let core_option =
        FamousDefs(&ctx.sema, ctx.sema.scope(type_ref.syntax())?.krate()).core_option_Option()?;

    let ty = ctx.sema.resolve_type(type_ref)?.as_adt();
    if matches!(ty, Some(hir::Adt::Enum(ret_type)) if ret_type == core_option) {
        // The return type is already wrapped in an Option
        cov_mark::hit!(wrap_return_type_in_option_simple_return_type_already_option);
        return None;
    }

    acc.add(
        AssistId("wrap_return_type_in_option", AssistKind::RefactorRewrite),
        "Wrap return type in Option",
        type_ref.syntax().text_range(),
        |edit| {
            let new_option_ty = option_type(ctx, &core_option, type_ref).clone_for_update();
            let body = edit.make_mut(ast::Expr::BlockExpr(body));

            let mut exprs_to_wrap = Vec::new();
            let tail_cb = &mut |e: &_| tail_cb_impl(&mut exprs_to_wrap, e);
            walk_expr(&body, &mut |expr| {
                if let Expr::ReturnExpr(ret_expr) = expr {
                    if let Some(ret_expr_arg) = &ret_expr.expr() {
                        for_each_tail_expr(ret_expr_arg, tail_cb);
                    }
                }
            });
            for_each_tail_expr(&body, tail_cb);

            for ret_expr_arg in exprs_to_wrap {
                let some_wrapped = make::expr_call(
                    make::expr_path(make::ext::ident_path("Some")),
                    make::arg_list(iter::once(ret_expr_arg.clone())),
                )
                .clone_for_update();
                ted::replace(ret_expr_arg.syntax(), some_wrapped.syntax());
            }

            let old_option_ty = edit.make_mut(type_ref.clone());
            ted::replace(old_option_ty.syntax(), new_option_ty.syntax());
        },
    )
}

fn option_type(
    ctx: &AssistContext<'_>,
    core_option: &hir::Enum,
    ret_type: &ast::Type,
) -> ast::Type {
    // Try to find an Option<T> type alias in the current scope (shadowing the default).
    let option_path = hir::ModPath::from_segments(
        hir::PathKind::Plain,
        iter::once(hir::Name::new_symbol_root(hir::sym::Option.clone())),
    );
    let alias = ctx.sema.resolve_mod_path(ret_type.syntax(), &option_path).and_then(|def| {
        def.filter_map(|def| match def.as_module_def()? {
            hir::ModuleDef::TypeAlias(alias) => {
                let enum_ty = alias.ty(ctx.db()).as_adt()?.as_enum()?;
                (&enum_ty == core_option).then_some(alias)
            }
            _ => None,
        })
        .find_map(|alias| {
            let mut inserted_ret_type = false;
            let generic_params = alias
                .source(ctx.db())?
                .value
                .generic_param_list()?
                .generic_params()
                .map(|param| match param {
                    // Replace the very first type parameter with the functions return type.
                    ast::GenericParam::TypeParam(_) if !inserted_ret_type => {
                        inserted_ret_type = true;
                        ret_type.to_smolstr()
                    }
                    ast::GenericParam::LifetimeParam(_) => make::lifetime("'_").to_smolstr(),
                    _ => make::ty_placeholder().to_smolstr(),
                })
                .join(", ");

            let name = alias.name(ctx.db());
            let name = name.as_str();
            Some(make::ty(&format!("{name}<{generic_params}>")))
        })
    });
    // If there is no applicable alias in scope use the default Option type.
    alias.unwrap_or_else(|| make::ext::ty_option(ret_type.clone()))
}

fn tail_cb_impl(acc: &mut Vec<ast::Expr>, e: &ast::Expr) {
    match e {
        Expr::BreakExpr(break_expr) => {
            if let Some(break_expr_arg) = break_expr.expr() {
                for_each_tail_expr(&break_expr_arg, &mut |e| tail_cb_impl(acc, e))
            }
        }
        Expr::ReturnExpr(_) => {
            // all return expressions have already been handled by the walk loop
        }
        e => acc.push(e.clone()),
    }
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_assist, check_assist_not_applicable};

    use super::*;

    #[test]
    fn wrap_return_type_in_option_simple() {
        check_assist(
            wrap_return_type_in_option,
            r#"
//- minicore: option
fn foo() -> i3$02 {
    let test = "test";
    return 42i32;
}
"#,
            r#"
fn foo() -> Option<i32> {
    let test = "test";
    return Some(42i32);
}
"#,
        );
    }

    #[test]
    fn wrap_return_type_break_split_tail() {
        check_assist(
            wrap_return_type_in_option,
            r#"
//- minicore: option
fn foo() -> i3$02 {
    loop {
        break if true {
            1
        } else {
            0
        };
    }
}
"#,
            r#"
fn foo() -> Option<i32> {
    loop {
        break if true {
            Some(1)
        } else {
            Some(0)
        };
    }
}
"#,
        );
    }

    #[test]
    fn wrap_return_type_in_option_simple_closure() {
        check_assist(
            wrap_return_type_in_option,
            r#"
//- minicore: option
fn foo() {
    || -> i32$0 {
        let test = "test";
        return 42i32;
    };
}
"#,
            r#"
fn foo() {
    || -> Option<i32> {
        let test = "test";
        return Some(42i32);
    };
}
"#,
        );
    }

    #[test]
    fn wrap_return_type_in_option_simple_return_type_bad_cursor() {
        check_assist_not_applicable(
            wrap_return_type_in_option,
            r#"
//- minicore: option
fn foo() -> i32 {
    let test = "test";$0
    return 42i32;
}
"#,
        );
    }

    #[test]
    fn wrap_return_type_in_option_simple_return_type_bad_cursor_closure() {
        check_assist_not_applicable(
            wrap_return_type_in_option,
            r#"
//- minicore: option
fn foo() {
    || -> i32 {
        let test = "test";$0
        return 42i32;
    };
}
"#,
        );
    }

    #[test]
    fn wrap_return_type_in_option_closure_non_block() {
        check_assist_not_applicable(
            wrap_return_type_in_option,
            r#"
//- minicore: option
fn foo() { || -> i$032 3; }
"#,
        );
    }

    #[test]
    fn wrap_return_type_in_option_simple_return_type_already_option_std() {
        check_assist_not_applicable(
            wrap_return_type_in_option,
            r#"
//- minicore: option
fn foo() -> core::option::Option<i32$0> {
    let test = "test";
    return 42i32;
}
"#,
        );
    }

    #[test]
    fn wrap_return_type_in_option_simple_return_type_already_option() {
        cov_mark::check!(wrap_return_type_in_option_simple_return_type_already_option);
        check_assist_not_applicable(
            wrap_return_type_in_option,
            r#"
//- minicore: option
fn foo() -> Option<i32$0> {
    let test = "test";
    return 42i32;
}
"#,
        );
    }

    #[test]
    fn wrap_return_type_in_option_simple_return_type_already_option_closure() {
        check_assist_not_applicable(
            wrap_return_type_in_option,
            r#"
//- minicore: option
fn foo() {
    || -> Option<i32$0, String> {
        let test = "test";
        return 42i32;
    };
}
"#,
        );
    }

    #[test]
    fn wrap_return_type_in_option_simple_with_cursor() {
        check_assist(
            wrap_return_type_in_option,
            r#"
//- minicore: option
fn foo() -> $0i32 {
    let test = "test";
    return 42i32;
}
"#,
            r#"
fn foo() -> Option<i32> {
    let test = "test";
    return Some(42i32);
}
"#,
        );
    }

    #[test]
    fn wrap_return_type_in_option_simple_with_tail() {
        check_assist(
            wrap_return_type_in_option,
            r#"
//- minicore: option
fn foo() ->$0 i32 {
    let test = "test";
    42i32
}
"#,
            r#"
fn foo() -> Option<i32> {
    let test = "test";
    Some(42i32)
}
"#,
        );
    }

    #[test]
    fn wrap_return_type_in_option_simple_with_tail_closure() {
        check_assist(
            wrap_return_type_in_option,
            r#"
//- minicore: option
fn foo() {
    || ->$0 i32 {
        let test = "test";
        42i32
    };
}
"#,
            r#"
fn foo() {
    || -> Option<i32> {
        let test = "test";
        Some(42i32)
    };
}
"#,
        );
    }

    #[test]
    fn wrap_return_type_in_option_simple_with_tail_only() {
        check_assist(
            wrap_return_type_in_option,
            r#"
//- minicore: option
fn foo() -> i32$0 { 42i32 }
"#,
            r#"
fn foo() -> Option<i32> { Some(42i32) }
"#,
        );
    }

    #[test]
    fn wrap_return_type_in_option_simple_with_tail_block_like() {
        check_assist(
            wrap_return_type_in_option,
            r#"
//- minicore: option
fn foo() -> i32$0 {
    if true {
        42i32
    } else {
        24i32
    }
}
"#,
            r#"
fn foo() -> Option<i32> {
    if true {
        Some(42i32)
    } else {
        Some(24i32)
    }
}
"#,
        );
    }

    #[test]
    fn wrap_return_type_in_option_simple_without_block_closure() {
        check_assist(
            wrap_return_type_in_option,
            r#"
//- minicore: option
fn foo() {
    || -> i32$0 {
        if true {
            42i32
        } else {
            24i32
        }
    };
}
"#,
            r#"
fn foo() {
    || -> Option<i32> {
        if true {
            Some(42i32)
        } else {
            Some(24i32)
        }
    };
}
"#,
        );
    }

    #[test]
    fn wrap_return_type_in_option_simple_with_nested_if() {
        check_assist(
            wrap_return_type_in_option,
            r#"
//- minicore: option
fn foo() -> i32$0 {
    if true {
        if false {
            1
        } else {
            2
        }
    } else {
        24i32
    }
}
"#,
            r#"
fn foo() -> Option<i32> {
    if true {
        if false {
            Some(1)
        } else {
            Some(2)
        }
    } else {
        Some(24i32)
    }
}
"#,
        );
    }

    #[test]
    fn wrap_return_type_in_option_simple_with_await() {
        check_assist(
            wrap_return_type_in_option,
            r#"
//- minicore: option
async fn foo() -> i$032 {
    if true {
        if false {
            1.await
        } else {
            2.await
        }
    } else {
        24i32.await
    }
}
"#,
            r#"
async fn foo() -> Option<i32> {
    if true {
        if false {
            Some(1.await)
        } else {
            Some(2.await)
        }
    } else {
        Some(24i32.await)
    }
}
"#,
        );
    }

    #[test]
    fn wrap_return_type_in_option_simple_with_array() {
        check_assist(
            wrap_return_type_in_option,
            r#"
//- minicore: option
fn foo() -> [i32;$0 3] { [1, 2, 3] }
"#,
            r#"
fn foo() -> Option<[i32; 3]> { Some([1, 2, 3]) }
"#,
        );
    }

    #[test]
    fn wrap_return_type_in_option_simple_with_cast() {
        check_assist(
            wrap_return_type_in_option,
            r#"
//- minicore: option
fn foo() -$0> i32 {
    if true {
        if false {
            1 as i32
        } else {
            2 as i32
        }
    } else {
        24 as i32
    }
}
"#,
            r#"
fn foo() -> Option<i32> {
    if true {
        if false {
            Some(1 as i32)
        } else {
            Some(2 as i32)
        }
    } else {
        Some(24 as i32)
    }
}
"#,
        );
    }

    #[test]
    fn wrap_return_type_in_option_simple_with_tail_block_like_match() {
        check_assist(
            wrap_return_type_in_option,
            r#"
//- minicore: option
fn foo() -> i32$0 {
    let my_var = 5;
    match my_var {
        5 => 42i32,
        _ => 24i32,
    }
}
"#,
            r#"
fn foo() -> Option<i32> {
    let my_var = 5;
    match my_var {
        5 => Some(42i32),
        _ => Some(24i32),
    }
}
"#,
        );
    }

    #[test]
    fn wrap_return_type_in_option_simple_with_loop_with_tail() {
        check_assist(
            wrap_return_type_in_option,
            r#"
//- minicore: option
fn foo() -> i32$0 {
    let my_var = 5;
    loop {
        println!("test");
        5
    }
    my_var
}
"#,
            r#"
fn foo() -> Option<i32> {
    let my_var = 5;
    loop {
        println!("test");
        5
    }
    Some(my_var)
}
"#,
        );
    }

    #[test]
    fn wrap_return_type_in_option_simple_with_loop_in_let_stmt() {
        check_assist(
            wrap_return_type_in_option,
            r#"
//- minicore: option
fn foo() -> i32$0 {
    let my_var = let x = loop {
        break 1;
    };
    my_var
}
"#,
            r#"
fn foo() -> Option<i32> {
    let my_var = let x = loop {
        break 1;
    };
    Some(my_var)
}
"#,
        );
    }

    #[test]
    fn wrap_return_type_in_option_simple_with_tail_block_like_match_return_expr() {
        check_assist(
            wrap_return_type_in_option,
            r#"
//- minicore: option
fn foo() -> i32$0 {
    let my_var = 5;
    let res = match my_var {
        5 => 42i32,
        _ => return 24i32,
    };
    res
}
"#,
            r#"
fn foo() -> Option<i32> {
    let my_var = 5;
    let res = match my_var {
        5 => 42i32,
        _ => return Some(24i32),
    };
    Some(res)
}
"#,
        );

        check_assist(
            wrap_return_type_in_option,
            r#"
//- minicore: option
fn foo() -> i32$0 {
    let my_var = 5;
    let res = if my_var == 5 {
        42i32
    } else {
        return 24i32;
    };
    res
}
"#,
            r#"
fn foo() -> Option<i32> {
    let my_var = 5;
    let res = if my_var == 5 {
        42i32
    } else {
        return Some(24i32);
    };
    Some(res)
}
"#,
        );
    }

    #[test]
    fn wrap_return_type_in_option_simple_with_tail_block_like_match_deeper() {
        check_assist(
            wrap_return_type_in_option,
            r#"
//- minicore: option
fn foo() -> i32$0 {
    let my_var = 5;
    match my_var {
        5 => {
            if true {
                42i32
            } else {
                25i32
            }
        },
        _ => {
            let test = "test";
            if test == "test" {
                return bar();
            }
            53i32
        },
    }
}
"#,
            r#"
fn foo() -> Option<i32> {
    let my_var = 5;
    match my_var {
        5 => {
            if true {
                Some(42i32)
            } else {
                Some(25i32)
            }
        },
        _ => {
            let test = "test";
            if test == "test" {
                return Some(bar());
            }
            Some(53i32)
        },
    }
}
"#,
        );
    }

    #[test]
    fn wrap_return_type_in_option_simple_with_tail_block_like_early_return() {
        check_assist(
            wrap_return_type_in_option,
            r#"
//- minicore: option
fn foo() -> i$032 {
    let test = "test";
    if test == "test" {
        return 24i32;
    }
    53i32
}
"#,
            r#"
fn foo() -> Option<i32> {
    let test = "test";
    if test == "test" {
        return Some(24i32);
    }
    Some(53i32)
}
"#,
        );
    }

    #[test]
    fn wrap_return_in_tail_position() {
        check_assist(
            wrap_return_type_in_option,
            r#"
//- minicore: option
fn foo(num: i32) -> $0i32 {
    return num
}
"#,
            r#"
fn foo(num: i32) -> Option<i32> {
    return Some(num)
}
"#,
        );
    }

    #[test]
    fn wrap_return_type_in_option_simple_with_closure() {
        check_assist(
            wrap_return_type_in_option,
            r#"
//- minicore: option
fn foo(the_field: u32) ->$0 u32 {
    let true_closure = || { return true; };
    if the_field < 5 {
        let mut i = 0;
        if true_closure() {
            return 99;
        } else {
            return 0;
        }
    }
    the_field
}
"#,
            r#"
fn foo(the_field: u32) -> Option<u32> {
    let true_closure = || { return true; };
    if the_field < 5 {
        let mut i = 0;
        if true_closure() {
            return Some(99);
        } else {
            return Some(0);
        }
    }
    Some(the_field)
}
"#,
        );

        check_assist(
            wrap_return_type_in_option,
            r#"
//- minicore: option
fn foo(the_field: u32) -> u32$0 {
    let true_closure = || {
        return true;
    };
    if the_field < 5 {
        let mut i = 0;


        if true_closure() {
            return 99;
        } else {
            return 0;
        }
    }
    let t = None;

    t.unwrap_or_else(|| the_field)
}
"#,
            r#"
fn foo(the_field: u32) -> Option<u32> {
    let true_closure = || {
        return true;
    };
    if the_field < 5 {
        let mut i = 0;


        if true_closure() {
            return Some(99);
        } else {
            return Some(0);
        }
    }
    let t = None;

    Some(t.unwrap_or_else(|| the_field))
}
"#,
        );
    }

    #[test]
    fn wrap_return_type_in_option_simple_with_weird_forms() {
        check_assist(
            wrap_return_type_in_option,
            r#"
//- minicore: option
fn foo() -> i32$0 {
    let test = "test";
    if test == "test" {
        return 24i32;
    }
    let mut i = 0;
    loop {
        if i == 1 {
            break 55;
        }
        i += 1;
    }
}
"#,
            r#"
fn foo() -> Option<i32> {
    let test = "test";
    if test == "test" {
        return Some(24i32);
    }
    let mut i = 0;
    loop {
        if i == 1 {
            break Some(55);
        }
        i += 1;
    }
}
"#,
        );

        check_assist(
            wrap_return_type_in_option,
            r#"
//- minicore: option
fn foo(the_field: u32) -> u32$0 {
    if the_field < 5 {
        let mut i = 0;
        loop {
            if i > 5 {
                return 55u32;
            }
            i += 3;
        }
        match i {
            5 => return 99,
            _ => return 0,
        };
    }
    the_field
}
"#,
            r#"
fn foo(the_field: u32) -> Option<u32> {
    if the_field < 5 {
        let mut i = 0;
        loop {
            if i > 5 {
                return Some(55u32);
            }
            i += 3;
        }
        match i {
            5 => return Some(99),
            _ => return Some(0),
        };
    }
    Some(the_field)
}
"#,
        );

        check_assist(
            wrap_return_type_in_option,
            r#"
//- minicore: option
fn foo(the_field: u32) -> u3$02 {
    if the_field < 5 {
        let mut i = 0;
        match i {
            5 => return 99,
            _ => return 0,
        }
    }
    the_field
}
"#,
            r#"
fn foo(the_field: u32) -> Option<u32> {
    if the_field < 5 {
        let mut i = 0;
        match i {
            5 => return Some(99),
            _ => return Some(0),
        }
    }
    Some(the_field)
}
"#,
        );

        check_assist(
            wrap_return_type_in_option,
            r#"
//- minicore: option
fn foo(the_field: u32) -> u32$0 {
    if the_field < 5 {
        let mut i = 0;
        if i == 5 {
            return 99
        } else {
            return 0
        }
    }
    the_field
}
"#,
            r#"
fn foo(the_field: u32) -> Option<u32> {
    if the_field < 5 {
        let mut i = 0;
        if i == 5 {
            return Some(99)
        } else {
            return Some(0)
        }
    }
    Some(the_field)
}
"#,
        );

        check_assist(
            wrap_return_type_in_option,
            r#"
//- minicore: option
fn foo(the_field: u32) -> $0u32 {
    if the_field < 5 {
        let mut i = 0;
        if i == 5 {
            return 99;
        } else {
            return 0;
        }
    }
    the_field
}
"#,
            r#"
fn foo(the_field: u32) -> Option<u32> {
    if the_field < 5 {
        let mut i = 0;
        if i == 5 {
            return Some(99);
        } else {
            return Some(0);
        }
    }
    Some(the_field)
}
"#,
        );
    }

    #[test]
    fn wrap_return_type_in_local_option_type() {
        check_assist(
            wrap_return_type_in_option,
            r#"
//- minicore: option
type Option<T> = core::option::Option<T>;

fn foo() -> i3$02 {
    return 42i32;
}
"#,
            r#"
type Option<T> = core::option::Option<T>;

fn foo() -> Option<i32> {
    return Some(42i32);
}
"#,
        );

        check_assist(
            wrap_return_type_in_option,
            r#"
//- minicore: option
type Option2<T> = core::option::Option<T>;

fn foo() -> i3$02 {
    return 42i32;
}
"#,
            r#"
type Option2<T> = core::option::Option<T>;

fn foo() -> Option<i32> {
    return Some(42i32);
}
"#,
        );
    }

    #[test]
    fn wrap_return_type_in_imported_local_option_type() {
        check_assist(
            wrap_return_type_in_option,
            r#"
//- minicore: option
mod some_module {
    pub type Option<T> = core::option::Option<T>;
}

use some_module::Option;

fn foo() -> i3$02 {
    return 42i32;
}
"#,
            r#"
mod some_module {
    pub type Option<T> = core::option::Option<T>;
}

use some_module::Option;

fn foo() -> Option<i32> {
    return Some(42i32);
}
"#,
        );

        check_assist(
            wrap_return_type_in_option,
            r#"
//- minicore: option
mod some_module {
    pub type Option<T> = core::option::Option<T>;
}

use some_module::*;

fn foo() -> i3$02 {
    return 42i32;
}
"#,
            r#"
mod some_module {
    pub type Option<T> = core::option::Option<T>;
}

use some_module::*;

fn foo() -> Option<i32> {
    return Some(42i32);
}
"#,
        );
    }

    #[test]
    fn wrap_return_type_in_local_option_type_from_function_body() {
        check_assist(
            wrap_return_type_in_option,
            r#"
//- minicore: option
fn foo() -> i3$02 {
    type Option<T> = core::option::Option<T>;
    0
}
"#,
            r#"
fn foo() -> Option<i32> {
    type Option<T> = core::option::Option<T>;
    Some(0)
}
"#,
        );
    }

    #[test]
    fn wrap_return_type_in_local_option_type_already_using_alias() {
        check_assist_not_applicable(
            wrap_return_type_in_option,
            r#"
//- minicore: option
pub type Option<T> = core::option::Option<T>;

fn foo() -> Option<i3$02> {
    return Some(42i32);
}
"#,
        );
    }
}
