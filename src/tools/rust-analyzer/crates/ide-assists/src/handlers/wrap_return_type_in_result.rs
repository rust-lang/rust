use std::iter;

use ide_db::{
    famous_defs::FamousDefs,
    syntax_helpers::node_ext::{for_each_tail_expr, walk_expr},
};
use syntax::{
    ast::{self, make, Expr},
    match_ast, ted, AstNode,
};

use crate::{AssistContext, AssistId, AssistKind, Assists};

// Assist: wrap_return_type_in_result
//
// Wrap the function's return type into Result.
//
// ```
// # //- minicore: result
// fn foo() -> i32$0 { 42i32 }
// ```
// ->
// ```
// fn foo() -> Result<i32, ${0:_}> { Ok(42i32) }
// ```
pub(crate) fn wrap_return_type_in_result(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
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
    let ty = ctx.sema.resolve_type(type_ref)?.as_adt();
    let result_enum =
        FamousDefs(&ctx.sema, ctx.sema.scope(type_ref.syntax())?.krate()).core_result_Result()?;

    if matches!(ty, Some(hir::Adt::Enum(ret_type)) if ret_type == result_enum) {
        cov_mark::hit!(wrap_return_type_in_result_simple_return_type_already_result);
        return None;
    }

    acc.add(
        AssistId("wrap_return_type_in_result", AssistKind::RefactorRewrite),
        "Wrap return type in Result",
        type_ref.syntax().text_range(),
        |edit| {
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
                let ok_wrapped = make::expr_call(
                    make::expr_path(make::ext::ident_path("Ok")),
                    make::arg_list(iter::once(ret_expr_arg.clone())),
                )
                .clone_for_update();
                ted::replace(ret_expr_arg.syntax(), ok_wrapped.syntax());
            }

            let new_result_ty =
                make::ext::ty_result(type_ref.clone(), make::ty_placeholder()).clone_for_update();
            let old_result_ty = edit.make_mut(type_ref.clone());

            ted::replace(old_result_ty.syntax(), new_result_ty.syntax());

            if let Some(cap) = ctx.config.snippet_cap {
                let generic_args = new_result_ty
                    .syntax()
                    .descendants()
                    .find_map(ast::GenericArgList::cast)
                    .unwrap();
                edit.add_placeholder_snippet(cap, generic_args.generic_args().last().unwrap());
            }
        },
    )
}

fn tail_cb_impl(acc: &mut Vec<ast::Expr>, e: &ast::Expr) {
    match e {
        Expr::BreakExpr(break_expr) => {
            if let Some(break_expr_arg) = break_expr.expr() {
                for_each_tail_expr(&break_expr_arg, &mut |e| tail_cb_impl(acc, e))
            }
        }
        Expr::ReturnExpr(ret_expr) => {
            if let Some(ret_expr_arg) = &ret_expr.expr() {
                for_each_tail_expr(ret_expr_arg, &mut |e| tail_cb_impl(acc, e));
            }
        }
        e => acc.push(e.clone()),
    }
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_assist, check_assist_not_applicable};

    use super::*;

    #[test]
    fn wrap_return_type_in_result_simple() {
        check_assist(
            wrap_return_type_in_result,
            r#"
//- minicore: result
fn foo() -> i3$02 {
    let test = "test";
    return 42i32;
}
"#,
            r#"
fn foo() -> Result<i32, ${0:_}> {
    let test = "test";
    return Ok(42i32);
}
"#,
        );
    }

    #[test]
    fn wrap_return_type_break_split_tail() {
        check_assist(
            wrap_return_type_in_result,
            r#"
//- minicore: result
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
fn foo() -> Result<i32, ${0:_}> {
    loop {
        break if true {
            Ok(1)
        } else {
            Ok(0)
        };
    }
}
"#,
        );
    }

    #[test]
    fn wrap_return_type_in_result_simple_closure() {
        check_assist(
            wrap_return_type_in_result,
            r#"
//- minicore: result
fn foo() {
    || -> i32$0 {
        let test = "test";
        return 42i32;
    };
}
"#,
            r#"
fn foo() {
    || -> Result<i32, ${0:_}> {
        let test = "test";
        return Ok(42i32);
    };
}
"#,
        );
    }

    #[test]
    fn wrap_return_type_in_result_simple_return_type_bad_cursor() {
        check_assist_not_applicable(
            wrap_return_type_in_result,
            r#"
//- minicore: result
fn foo() -> i32 {
    let test = "test";$0
    return 42i32;
}
"#,
        );
    }

    #[test]
    fn wrap_return_type_in_result_simple_return_type_bad_cursor_closure() {
        check_assist_not_applicable(
            wrap_return_type_in_result,
            r#"
//- minicore: result
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
    fn wrap_return_type_in_result_closure_non_block() {
        check_assist_not_applicable(
            wrap_return_type_in_result,
            r#"
//- minicore: result
fn foo() { || -> i$032 3; }
"#,
        );
    }

    #[test]
    fn wrap_return_type_in_result_simple_return_type_already_result_std() {
        check_assist_not_applicable(
            wrap_return_type_in_result,
            r#"
//- minicore: result
fn foo() -> core::result::Result<i32$0, String> {
    let test = "test";
    return 42i32;
}
"#,
        );
    }

    #[test]
    fn wrap_return_type_in_result_simple_return_type_already_result() {
        cov_mark::check!(wrap_return_type_in_result_simple_return_type_already_result);
        check_assist_not_applicable(
            wrap_return_type_in_result,
            r#"
//- minicore: result
fn foo() -> Result<i32$0, String> {
    let test = "test";
    return 42i32;
}
"#,
        );
    }

    #[test]
    fn wrap_return_type_in_result_simple_return_type_already_result_closure() {
        check_assist_not_applicable(
            wrap_return_type_in_result,
            r#"
//- minicore: result
fn foo() {
    || -> Result<i32$0, String> {
        let test = "test";
        return 42i32;
    };
}
"#,
        );
    }

    #[test]
    fn wrap_return_type_in_result_simple_with_cursor() {
        check_assist(
            wrap_return_type_in_result,
            r#"
//- minicore: result
fn foo() -> $0i32 {
    let test = "test";
    return 42i32;
}
"#,
            r#"
fn foo() -> Result<i32, ${0:_}> {
    let test = "test";
    return Ok(42i32);
}
"#,
        );
    }

    #[test]
    fn wrap_return_type_in_result_simple_with_tail() {
        check_assist(
            wrap_return_type_in_result,
            r#"
//- minicore: result
fn foo() ->$0 i32 {
    let test = "test";
    42i32
}
"#,
            r#"
fn foo() -> Result<i32, ${0:_}> {
    let test = "test";
    Ok(42i32)
}
"#,
        );
    }

    #[test]
    fn wrap_return_type_in_result_simple_with_tail_closure() {
        check_assist(
            wrap_return_type_in_result,
            r#"
//- minicore: result
fn foo() {
    || ->$0 i32 {
        let test = "test";
        42i32
    };
}
"#,
            r#"
fn foo() {
    || -> Result<i32, ${0:_}> {
        let test = "test";
        Ok(42i32)
    };
}
"#,
        );
    }

    #[test]
    fn wrap_return_type_in_result_simple_with_tail_only() {
        check_assist(
            wrap_return_type_in_result,
            r#"
//- minicore: result
fn foo() -> i32$0 { 42i32 }
"#,
            r#"
fn foo() -> Result<i32, ${0:_}> { Ok(42i32) }
"#,
        );
    }

    #[test]
    fn wrap_return_type_in_result_simple_with_tail_block_like() {
        check_assist(
            wrap_return_type_in_result,
            r#"
//- minicore: result
fn foo() -> i32$0 {
    if true {
        42i32
    } else {
        24i32
    }
}
"#,
            r#"
fn foo() -> Result<i32, ${0:_}> {
    if true {
        Ok(42i32)
    } else {
        Ok(24i32)
    }
}
"#,
        );
    }

    #[test]
    fn wrap_return_type_in_result_simple_without_block_closure() {
        check_assist(
            wrap_return_type_in_result,
            r#"
//- minicore: result
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
    || -> Result<i32, ${0:_}> {
        if true {
            Ok(42i32)
        } else {
            Ok(24i32)
        }
    };
}
"#,
        );
    }

    #[test]
    fn wrap_return_type_in_result_simple_with_nested_if() {
        check_assist(
            wrap_return_type_in_result,
            r#"
//- minicore: result
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
fn foo() -> Result<i32, ${0:_}> {
    if true {
        if false {
            Ok(1)
        } else {
            Ok(2)
        }
    } else {
        Ok(24i32)
    }
}
"#,
        );
    }

    #[test]
    fn wrap_return_type_in_result_simple_with_await() {
        check_assist(
            wrap_return_type_in_result,
            r#"
//- minicore: result
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
async fn foo() -> Result<i32, ${0:_}> {
    if true {
        if false {
            Ok(1.await)
        } else {
            Ok(2.await)
        }
    } else {
        Ok(24i32.await)
    }
}
"#,
        );
    }

    #[test]
    fn wrap_return_type_in_result_simple_with_array() {
        check_assist(
            wrap_return_type_in_result,
            r#"
//- minicore: result
fn foo() -> [i32;$0 3] { [1, 2, 3] }
"#,
            r#"
fn foo() -> Result<[i32; 3], ${0:_}> { Ok([1, 2, 3]) }
"#,
        );
    }

    #[test]
    fn wrap_return_type_in_result_simple_with_cast() {
        check_assist(
            wrap_return_type_in_result,
            r#"
//- minicore: result
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
fn foo() -> Result<i32, ${0:_}> {
    if true {
        if false {
            Ok(1 as i32)
        } else {
            Ok(2 as i32)
        }
    } else {
        Ok(24 as i32)
    }
}
"#,
        );
    }

    #[test]
    fn wrap_return_type_in_result_simple_with_tail_block_like_match() {
        check_assist(
            wrap_return_type_in_result,
            r#"
//- minicore: result
fn foo() -> i32$0 {
    let my_var = 5;
    match my_var {
        5 => 42i32,
        _ => 24i32,
    }
}
"#,
            r#"
fn foo() -> Result<i32, ${0:_}> {
    let my_var = 5;
    match my_var {
        5 => Ok(42i32),
        _ => Ok(24i32),
    }
}
"#,
        );
    }

    #[test]
    fn wrap_return_type_in_result_simple_with_loop_with_tail() {
        check_assist(
            wrap_return_type_in_result,
            r#"
//- minicore: result
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
fn foo() -> Result<i32, ${0:_}> {
    let my_var = 5;
    loop {
        println!("test");
        5
    }
    Ok(my_var)
}
"#,
        );
    }

    #[test]
    fn wrap_return_type_in_result_simple_with_loop_in_let_stmt() {
        check_assist(
            wrap_return_type_in_result,
            r#"
//- minicore: result
fn foo() -> i32$0 {
    let my_var = let x = loop {
        break 1;
    };
    my_var
}
"#,
            r#"
fn foo() -> Result<i32, ${0:_}> {
    let my_var = let x = loop {
        break 1;
    };
    Ok(my_var)
}
"#,
        );
    }

    #[test]
    fn wrap_return_type_in_result_simple_with_tail_block_like_match_return_expr() {
        check_assist(
            wrap_return_type_in_result,
            r#"
//- minicore: result
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
fn foo() -> Result<i32, ${0:_}> {
    let my_var = 5;
    let res = match my_var {
        5 => 42i32,
        _ => return Ok(24i32),
    };
    Ok(res)
}
"#,
        );

        check_assist(
            wrap_return_type_in_result,
            r#"
//- minicore: result
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
fn foo() -> Result<i32, ${0:_}> {
    let my_var = 5;
    let res = if my_var == 5 {
        42i32
    } else {
        return Ok(24i32);
    };
    Ok(res)
}
"#,
        );
    }

    #[test]
    fn wrap_return_type_in_result_simple_with_tail_block_like_match_deeper() {
        check_assist(
            wrap_return_type_in_result,
            r#"
//- minicore: result
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
fn foo() -> Result<i32, ${0:_}> {
    let my_var = 5;
    match my_var {
        5 => {
            if true {
                Ok(42i32)
            } else {
                Ok(25i32)
            }
        },
        _ => {
            let test = "test";
            if test == "test" {
                return Ok(bar());
            }
            Ok(53i32)
        },
    }
}
"#,
        );
    }

    #[test]
    fn wrap_return_type_in_result_simple_with_tail_block_like_early_return() {
        check_assist(
            wrap_return_type_in_result,
            r#"
//- minicore: result
fn foo() -> i$032 {
    let test = "test";
    if test == "test" {
        return 24i32;
    }
    53i32
}
"#,
            r#"
fn foo() -> Result<i32, ${0:_}> {
    let test = "test";
    if test == "test" {
        return Ok(24i32);
    }
    Ok(53i32)
}
"#,
        );
    }

    #[test]
    fn wrap_return_type_in_result_simple_with_closure() {
        check_assist(
            wrap_return_type_in_result,
            r#"
//- minicore: result
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
fn foo(the_field: u32) -> Result<u32, ${0:_}> {
    let true_closure = || { return true; };
    if the_field < 5 {
        let mut i = 0;
        if true_closure() {
            return Ok(99);
        } else {
            return Ok(0);
        }
    }
    Ok(the_field)
}
"#,
        );

        check_assist(
            wrap_return_type_in_result,
            r#"
//- minicore: result
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
fn foo(the_field: u32) -> Result<u32, ${0:_}> {
    let true_closure = || {
        return true;
    };
    if the_field < 5 {
        let mut i = 0;


        if true_closure() {
            return Ok(99);
        } else {
            return Ok(0);
        }
    }
    let t = None;

    Ok(t.unwrap_or_else(|| the_field))
}
"#,
        );
    }

    #[test]
    fn wrap_return_type_in_result_simple_with_weird_forms() {
        check_assist(
            wrap_return_type_in_result,
            r#"
//- minicore: result
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
fn foo() -> Result<i32, ${0:_}> {
    let test = "test";
    if test == "test" {
        return Ok(24i32);
    }
    let mut i = 0;
    loop {
        if i == 1 {
            break Ok(55);
        }
        i += 1;
    }
}
"#,
        );

        check_assist(
            wrap_return_type_in_result,
            r#"
//- minicore: result
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
fn foo(the_field: u32) -> Result<u32, ${0:_}> {
    if the_field < 5 {
        let mut i = 0;
        loop {
            if i > 5 {
                return Ok(55u32);
            }
            i += 3;
        }
        match i {
            5 => return Ok(99),
            _ => return Ok(0),
        };
    }
    Ok(the_field)
}
"#,
        );

        check_assist(
            wrap_return_type_in_result,
            r#"
//- minicore: result
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
fn foo(the_field: u32) -> Result<u32, ${0:_}> {
    if the_field < 5 {
        let mut i = 0;
        match i {
            5 => return Ok(99),
            _ => return Ok(0),
        }
    }
    Ok(the_field)
}
"#,
        );

        check_assist(
            wrap_return_type_in_result,
            r#"
//- minicore: result
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
fn foo(the_field: u32) -> Result<u32, ${0:_}> {
    if the_field < 5 {
        let mut i = 0;
        if i == 5 {
            return Ok(99)
        } else {
            return Ok(0)
        }
    }
    Ok(the_field)
}
"#,
        );

        check_assist(
            wrap_return_type_in_result,
            r#"
//- minicore: result
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
fn foo(the_field: u32) -> Result<u32, ${0:_}> {
    if the_field < 5 {
        let mut i = 0;
        if i == 5 {
            return Ok(99);
        } else {
            return Ok(0);
        }
    }
    Ok(the_field)
}
"#,
        );
    }
}
