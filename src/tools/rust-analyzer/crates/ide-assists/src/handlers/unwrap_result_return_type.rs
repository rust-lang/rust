use ide_db::{
    famous_defs::FamousDefs,
    syntax_helpers::node_ext::{for_each_tail_expr, walk_expr},
};
use itertools::Itertools;
use syntax::{
    ast::{self, Expr},
    match_ast, AstNode, NodeOrToken, SyntaxKind, TextRange,
};

use crate::{AssistContext, AssistId, AssistKind, Assists};

// Assist: unwrap_result_return_type
//
// Unwrap the function's return type.
//
// ```
// # //- minicore: result
// fn foo() -> Result<i32>$0 { Ok(42i32) }
// ```
// ->
// ```
// fn foo() -> i32 { 42i32 }
// ```
pub(crate) fn unwrap_result_return_type(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
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
    let Some(hir::Adt::Enum(ret_enum)) = ctx.sema.resolve_type(type_ref)?.as_adt() else { return None; };
    let result_enum =
        FamousDefs(&ctx.sema, ctx.sema.scope(type_ref.syntax())?.krate()).core_result_Result()?;
    if ret_enum != result_enum {
        return None;
    }

    let Some(ok_type) = unwrap_result_type(type_ref) else { return None; };

    acc.add(
        AssistId("unwrap_result_return_type", AssistKind::RefactorRewrite),
        "Unwrap Result return type",
        type_ref.syntax().text_range(),
        |builder| {
            let body = ast::Expr::BlockExpr(body);

            let mut exprs_to_unwrap = Vec::new();
            let tail_cb = &mut |e: &_| tail_cb_impl(&mut exprs_to_unwrap, e);
            walk_expr(&body, &mut |expr| {
                if let Expr::ReturnExpr(ret_expr) = expr {
                    if let Some(ret_expr_arg) = &ret_expr.expr() {
                        for_each_tail_expr(ret_expr_arg, tail_cb);
                    }
                }
            });
            for_each_tail_expr(&body, tail_cb);

            let is_unit_type = is_unit_type(&ok_type);
            if is_unit_type {
                let mut text_range = ret_type.syntax().text_range();

                if let Some(NodeOrToken::Token(token)) = ret_type.syntax().next_sibling_or_token() {
                    if token.kind() == SyntaxKind::WHITESPACE {
                        text_range = TextRange::new(text_range.start(), token.text_range().end());
                    }
                }

                builder.delete(text_range);
            } else {
                builder.replace(type_ref.syntax().text_range(), ok_type.syntax().text());
            }

            for ret_expr_arg in exprs_to_unwrap {
                let ret_expr_str = ret_expr_arg.to_string();
                if ret_expr_str.starts_with("Ok(") || ret_expr_str.starts_with("Err(") {
                    let arg_list = ret_expr_arg.syntax().children().find_map(ast::ArgList::cast);
                    if let Some(arg_list) = arg_list {
                        if is_unit_type {
                            match ret_expr_arg.syntax().prev_sibling_or_token() {
                                // Useful to delete the entire line without leaving trailing whitespaces
                                Some(whitespace) => {
                                    let new_range = TextRange::new(
                                        whitespace.text_range().start(),
                                        ret_expr_arg.syntax().text_range().end(),
                                    );
                                    builder.delete(new_range);
                                }
                                None => {
                                    builder.delete(ret_expr_arg.syntax().text_range());
                                }
                            }
                        } else {
                            builder.replace(
                                ret_expr_arg.syntax().text_range(),
                                arg_list.args().join(", "),
                            );
                        }
                    }
                }
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

// Tries to extract `T` from `Result<T, E>`.
fn unwrap_result_type(ty: &ast::Type) -> Option<ast::Type> {
    let ast::Type::PathType(path_ty) = ty else { return None; };
    let path = path_ty.path()?;
    let segment = path.first_segment()?;
    let generic_arg_list = segment.generic_arg_list()?;
    let generic_args: Vec<_> = generic_arg_list.generic_args().collect();
    let ast::GenericArg::TypeArg(ok_type) = generic_args.first()? else { return None; };
    ok_type.ty()
}

fn is_unit_type(ty: &ast::Type) -> bool {
    let ast::Type::TupleType(tuple) = ty else { return false };
    tuple.fields().next().is_none()
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_assist, check_assist_not_applicable};

    use super::*;

    #[test]
    fn unwrap_result_return_type_simple() {
        check_assist(
            unwrap_result_return_type,
            r#"
//- minicore: result
fn foo() -> Result<i3$02> {
    let test = "test";
    return Ok(42i32);
}
"#,
            r#"
fn foo() -> i32 {
    let test = "test";
    return 42i32;
}
"#,
        );
    }

    #[test]
    fn unwrap_result_return_type_unit_type() {
        check_assist(
            unwrap_result_return_type,
            r#"
//- minicore: result
fn foo() -> Result<(), Box<dyn Error$0>> {
    Ok(())
}
"#,
            r#"
fn foo() {
}
"#,
        );

        // Unformatted return type
        check_assist(
            unwrap_result_return_type,
            r#"
//- minicore: result
fn foo() -> Result<(), Box<dyn Error$0>>{
    Ok(())
}
"#,
            r#"
fn foo() {
}
"#,
        );
    }

    #[test]
    fn unwrap_result_return_type_ending_with_parent() {
        check_assist(
            unwrap_result_return_type,
            r#"
//- minicore: result
fn foo() -> Result<i32, Box<dyn Error$0>> {
    if true {
        Ok(42)
    } else {
        foo()
    }
}
"#,
            r#"
fn foo() -> i32 {
    if true {
        42
    } else {
        foo()
    }
}
"#,
        );
    }

    #[test]
    fn unwrap_return_type_break_split_tail() {
        check_assist(
            unwrap_result_return_type,
            r#"
//- minicore: result
fn foo() -> Result<i3$02, String> {
    loop {
        break if true {
            Ok(1)
        } else {
            Ok(0)
        };
    }
}
"#,
            r#"
fn foo() -> i32 {
    loop {
        break if true {
            1
        } else {
            0
        };
    }
}
"#,
        );
    }

    #[test]
    fn unwrap_result_return_type_simple_closure() {
        check_assist(
            unwrap_result_return_type,
            r#"
//- minicore: result
fn foo() {
    || -> Result<i32$0> {
        let test = "test";
        return Ok(42i32);
    };
}
"#,
            r#"
fn foo() {
    || -> i32 {
        let test = "test";
        return 42i32;
    };
}
"#,
        );
    }

    #[test]
    fn unwrap_result_return_type_simple_return_type_bad_cursor() {
        check_assist_not_applicable(
            unwrap_result_return_type,
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
    fn unwrap_result_return_type_simple_return_type_bad_cursor_closure() {
        check_assist_not_applicable(
            unwrap_result_return_type,
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
    fn unwrap_result_return_type_closure_non_block() {
        check_assist_not_applicable(
            unwrap_result_return_type,
            r#"
//- minicore: result
fn foo() { || -> i$032 3; }
"#,
        );
    }

    #[test]
    fn unwrap_result_return_type_simple_return_type_already_not_result_std() {
        check_assist_not_applicable(
            unwrap_result_return_type,
            r#"
//- minicore: result
fn foo() -> i32$0 {
    let test = "test";
    return 42i32;
}
"#,
        );
    }

    #[test]
    fn unwrap_result_return_type_simple_return_type_already_not_result_closure() {
        check_assist_not_applicable(
            unwrap_result_return_type,
            r#"
//- minicore: result
fn foo() {
    || -> i32$0 {
        let test = "test";
        return 42i32;
    };
}
"#,
        );
    }

    #[test]
    fn unwrap_result_return_type_simple_with_tail() {
        check_assist(
            unwrap_result_return_type,
            r#"
//- minicore: result
fn foo() ->$0 Result<i32> {
    let test = "test";
    Ok(42i32)
}
"#,
            r#"
fn foo() -> i32 {
    let test = "test";
    42i32
}
"#,
        );
    }

    #[test]
    fn unwrap_result_return_type_simple_with_tail_closure() {
        check_assist(
            unwrap_result_return_type,
            r#"
//- minicore: result
fn foo() {
    || ->$0 Result<i32, String> {
        let test = "test";
        Ok(42i32)
    };
}
"#,
            r#"
fn foo() {
    || -> i32 {
        let test = "test";
        42i32
    };
}
"#,
        );
    }

    #[test]
    fn unwrap_result_return_type_simple_with_tail_only() {
        check_assist(
            unwrap_result_return_type,
            r#"
//- minicore: result
fn foo() -> Result<i32$0> { Ok(42i32) }
"#,
            r#"
fn foo() -> i32 { 42i32 }
"#,
        );
    }

    #[test]
    fn unwrap_result_return_type_simple_with_tail_block_like() {
        check_assist(
            unwrap_result_return_type,
            r#"
//- minicore: result
fn foo() -> Result<i32>$0 {
    if true {
        Ok(42i32)
    } else {
        Ok(24i32)
    }
}
"#,
            r#"
fn foo() -> i32 {
    if true {
        42i32
    } else {
        24i32
    }
}
"#,
        );
    }

    #[test]
    fn unwrap_result_return_type_simple_without_block_closure() {
        check_assist(
            unwrap_result_return_type,
            r#"
//- minicore: result
fn foo() {
    || -> Result<i32, String>$0 {
        if true {
            Ok(42i32)
        } else {
            Ok(24i32)
        }
    };
}
"#,
            r#"
fn foo() {
    || -> i32 {
        if true {
            42i32
        } else {
            24i32
        }
    };
}
"#,
        );
    }

    #[test]
    fn unwrap_result_return_type_simple_with_nested_if() {
        check_assist(
            unwrap_result_return_type,
            r#"
//- minicore: result
fn foo() -> Result<i32>$0 {
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
            r#"
fn foo() -> i32 {
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
        );
    }

    #[test]
    fn unwrap_result_return_type_simple_with_await() {
        check_assist(
            unwrap_result_return_type,
            r#"
//- minicore: result
async fn foo() -> Result<i$032> {
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
            r#"
async fn foo() -> i32 {
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
        );
    }

    #[test]
    fn unwrap_result_return_type_simple_with_array() {
        check_assist(
            unwrap_result_return_type,
            r#"
//- minicore: result
fn foo() -> Result<[i32; 3]$0> { Ok([1, 2, 3]) }
"#,
            r#"
fn foo() -> [i32; 3] { [1, 2, 3] }
"#,
        );
    }

    #[test]
    fn unwrap_result_return_type_simple_with_cast() {
        check_assist(
            unwrap_result_return_type,
            r#"
//- minicore: result
fn foo() -$0> Result<i32> {
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
            r#"
fn foo() -> i32 {
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
        );
    }

    #[test]
    fn unwrap_result_return_type_simple_with_tail_block_like_match() {
        check_assist(
            unwrap_result_return_type,
            r#"
//- minicore: result
fn foo() -> Result<i32$0> {
    let my_var = 5;
    match my_var {
        5 => Ok(42i32),
        _ => Ok(24i32),
    }
}
"#,
            r#"
fn foo() -> i32 {
    let my_var = 5;
    match my_var {
        5 => 42i32,
        _ => 24i32,
    }
}
"#,
        );
    }

    #[test]
    fn unwrap_result_return_type_simple_with_loop_with_tail() {
        check_assist(
            unwrap_result_return_type,
            r#"
//- minicore: result
fn foo() -> Result<i32$0> {
    let my_var = 5;
    loop {
        println!("test");
        5
    }
    Ok(my_var)
}
"#,
            r#"
fn foo() -> i32 {
    let my_var = 5;
    loop {
        println!("test");
        5
    }
    my_var
}
"#,
        );
    }

    #[test]
    fn unwrap_result_return_type_simple_with_loop_in_let_stmt() {
        check_assist(
            unwrap_result_return_type,
            r#"
//- minicore: result
fn foo() -> Result<i32$0> {
    let my_var = let x = loop {
        break 1;
    };
    Ok(my_var)
}
"#,
            r#"
fn foo() -> i32 {
    let my_var = let x = loop {
        break 1;
    };
    my_var
}
"#,
        );
    }

    #[test]
    fn unwrap_result_return_type_simple_with_tail_block_like_match_return_expr() {
        check_assist(
            unwrap_result_return_type,
            r#"
//- minicore: result
fn foo() -> Result<i32>$0 {
    let my_var = 5;
    let res = match my_var {
        5 => 42i32,
        _ => return Ok(24i32),
    };
    Ok(res)
}
"#,
            r#"
fn foo() -> i32 {
    let my_var = 5;
    let res = match my_var {
        5 => 42i32,
        _ => return 24i32,
    };
    res
}
"#,
        );

        check_assist(
            unwrap_result_return_type,
            r#"
//- minicore: result
fn foo() -> Result<i32$0> {
    let my_var = 5;
    let res = if my_var == 5 {
        42i32
    } else {
        return Ok(24i32);
    };
    Ok(res)
}
"#,
            r#"
fn foo() -> i32 {
    let my_var = 5;
    let res = if my_var == 5 {
        42i32
    } else {
        return 24i32;
    };
    res
}
"#,
        );
    }

    #[test]
    fn unwrap_result_return_type_simple_with_tail_block_like_match_deeper() {
        check_assist(
            unwrap_result_return_type,
            r#"
//- minicore: result
fn foo() -> Result<i32$0> {
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
            r#"
fn foo() -> i32 {
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
        );
    }

    #[test]
    fn unwrap_result_return_type_simple_with_tail_block_like_early_return() {
        check_assist(
            unwrap_result_return_type,
            r#"
//- minicore: result
fn foo() -> Result<i32$0> {
    let test = "test";
    if test == "test" {
        return Ok(24i32);
    }
    Ok(53i32)
}
"#,
            r#"
fn foo() -> i32 {
    let test = "test";
    if test == "test" {
        return 24i32;
    }
    53i32
}
"#,
        );
    }

    #[test]
    fn unwrap_result_return_type_simple_with_closure() {
        check_assist(
            unwrap_result_return_type,
            r#"
//- minicore: result
fn foo(the_field: u32) -> Result<u32$0> {
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
            r#"
fn foo(the_field: u32) -> u32 {
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
        );

        check_assist(
            unwrap_result_return_type,
            r#"
//- minicore: result
fn foo(the_field: u32) -> Result<u32$0> {
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
            r#"
fn foo(the_field: u32) -> u32 {
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
        );
    }

    #[test]
    fn unwrap_result_return_type_simple_with_weird_forms() {
        check_assist(
            unwrap_result_return_type,
            r#"
//- minicore: result
fn foo() -> Result<i32$0> {
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
            r#"
fn foo() -> i32 {
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
        );

        check_assist(
            unwrap_result_return_type,
            r#"
//- minicore: result
fn foo(the_field: u32) -> Result<u32$0> {
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
            r#"
fn foo(the_field: u32) -> u32 {
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
        );

        check_assist(
            unwrap_result_return_type,
            r#"
//- minicore: result
fn foo(the_field: u32) -> Result<u32$0> {
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
            r#"
fn foo(the_field: u32) -> u32 {
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
        );

        check_assist(
            unwrap_result_return_type,
            r#"
//- minicore: result
fn foo(the_field: u32) -> Result<u32$0> {
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
            r#"
fn foo(the_field: u32) -> u32 {
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
        );

        check_assist(
            unwrap_result_return_type,
            r#"
//- minicore: result
fn foo(the_field: u32) -> Result<u3$02> {
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
            r#"
fn foo(the_field: u32) -> u32 {
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
        );
    }

    #[test]
    fn unwrap_result_return_type_nested_type() {
        check_assist(
            unwrap_result_return_type,
            r#"
//- minicore: result, option
fn foo() -> Result<Option<i32$0>, ()> {
    Ok(Some(42))
}
"#,
            r#"
fn foo() -> Option<i32> {
    Some(42)
}
"#,
        );

        check_assist(
            unwrap_result_return_type,
            r#"
//- minicore: result, option
fn foo() -> Result<Option<Result<i32$0, ()>>, ()> {
    Ok(None)
}
"#,
            r#"
fn foo() -> Option<Result<i32, ()>> {
    None
}
"#,
        );

        check_assist(
            unwrap_result_return_type,
            r#"
//- minicore: result, option, iterators
fn foo() -> Result<impl Iterator<Item = i32>$0, ()> {
    Ok(Some(42).into_iter())
}
"#,
            r#"
fn foo() -> impl Iterator<Item = i32> {
    Some(42).into_iter()
}
"#,
        );
    }
}
