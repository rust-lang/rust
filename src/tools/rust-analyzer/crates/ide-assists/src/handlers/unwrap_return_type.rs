use either::Either;
use ide_db::{
    famous_defs::FamousDefs,
    syntax_helpers::node_ext::{for_each_tail_expr, walk_expr},
};
use syntax::{
    AstNode, NodeOrToken, SyntaxKind,
    ast::{self, HasArgList, HasGenericArgs, syntax_factory::SyntaxFactory},
    match_ast,
};

use crate::{AssistContext, AssistId, Assists};

// Assist: unwrap_option_return_type
//
// Unwrap the function's return type.
//
// ```
// # //- minicore: option
// fn foo() -> Option<i32>$0 { Some(42i32) }
// ```
// ->
// ```
// fn foo() -> i32 { 42i32 }
// ```

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

pub(crate) fn unwrap_return_type(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    let ret_type = ctx.find_node_at_offset::<ast::RetType>()?;
    let parent = ret_type.syntax().parent()?;
    let body_expr = match_ast! {
        match parent {
            ast::Fn(func) => func.body()?.into(),
            ast::ClosureExpr(closure) => match closure.body()? {
                ast::Expr::BlockExpr(block) => block.into(),
                // closures require a block when a return type is specified
                _ => return None,
            },
            _ => return None,
        }
    };

    let type_ref = &ret_type.ty()?;
    let Some(hir::Adt::Enum(ret_enum)) = ctx.sema.resolve_type(type_ref)?.as_adt() else {
        return None;
    };

    let famous_defs = FamousDefs(&ctx.sema, ctx.sema.scope(type_ref.syntax())?.krate());

    let kind = UnwrapperKind::ALL
        .iter()
        .find(|k| matches!(k.core_type(&famous_defs), Some(core_type) if ret_enum == core_type))?;

    let happy_type = extract_wrapped_type(type_ref)?;

    acc.add(kind.assist_id(), kind.label(), type_ref.syntax().text_range(), |builder| {
        let mut editor = builder.make_editor(&parent);
        let make = SyntaxFactory::with_mappings();

        let mut exprs_to_unwrap = Vec::new();
        let tail_cb = &mut |e: &_| tail_cb_impl(&mut exprs_to_unwrap, e);
        walk_expr(&body_expr, &mut |expr| {
            if let ast::Expr::ReturnExpr(ret_expr) = expr
                && let Some(ret_expr_arg) = &ret_expr.expr()
            {
                for_each_tail_expr(ret_expr_arg, tail_cb);
            }
        });
        for_each_tail_expr(&body_expr, tail_cb);

        let is_unit_type = is_unit_type(&happy_type);
        if is_unit_type {
            if let Some(NodeOrToken::Token(token)) = ret_type.syntax().next_sibling_or_token()
                && token.kind() == SyntaxKind::WHITESPACE
            {
                editor.delete(token);
            }

            editor.delete(ret_type.syntax());
        } else {
            editor.replace(type_ref.syntax(), happy_type.syntax());
        }

        let mut final_placeholder = None;
        for tail_expr in exprs_to_unwrap {
            match &tail_expr {
                ast::Expr::CallExpr(call_expr) => {
                    let ast::Expr::PathExpr(path_expr) = call_expr.expr().unwrap() else {
                        continue;
                    };

                    let path_str = path_expr.path().unwrap().to_string();
                    let needs_replacing = match kind {
                        UnwrapperKind::Option => path_str == "Some",
                        UnwrapperKind::Result => path_str == "Ok" || path_str == "Err",
                    };

                    if !needs_replacing {
                        continue;
                    }

                    let arg_list = call_expr.arg_list().unwrap();
                    if is_unit_type {
                        let tail_parent = tail_expr
                            .syntax()
                            .parent()
                            .and_then(Either::<ast::ReturnExpr, ast::StmtList>::cast)
                            .unwrap();
                        match tail_parent {
                            Either::Left(ret_expr) => {
                                editor.replace(ret_expr.syntax(), make.expr_return(None).syntax())
                            }
                            Either::Right(stmt_list) => {
                                let new_block = if stmt_list.statements().next().is_none() {
                                    make.expr_empty_block()
                                } else {
                                    make.block_expr(stmt_list.statements(), None)
                                };
                                editor.replace(
                                    stmt_list.syntax(),
                                    new_block.stmt_list().unwrap().syntax(),
                                );
                            }
                        }
                    } else if let Some(first_arg) = arg_list.args().next() {
                        editor.replace(tail_expr.syntax(), first_arg.syntax());
                    }
                }
                ast::Expr::PathExpr(path_expr) => {
                    let UnwrapperKind::Option = kind else {
                        continue;
                    };

                    if path_expr.path().unwrap().to_string() != "None" {
                        continue;
                    }

                    let new_tail_expr = make.expr_unit();
                    editor.replace(path_expr.syntax(), new_tail_expr.syntax());
                    if let Some(cap) = ctx.config.snippet_cap {
                        editor.add_annotation(
                            new_tail_expr.syntax(),
                            builder.make_placeholder_snippet(cap),
                        );

                        final_placeholder = Some(new_tail_expr);
                    }
                }
                _ => (),
            }
        }

        if let Some(cap) = ctx.config.snippet_cap
            && let Some(final_placeholder) = final_placeholder
        {
            editor.add_annotation(final_placeholder.syntax(), builder.make_tabstop_after(cap));
        }

        editor.add_mappings(make.finish_with_mappings());
        builder.add_file_edits(ctx.vfs_file_id(), editor);
    })
}

enum UnwrapperKind {
    Option,
    Result,
}

impl UnwrapperKind {
    const ALL: &'static [UnwrapperKind] = &[UnwrapperKind::Option, UnwrapperKind::Result];

    fn assist_id(&self) -> AssistId {
        let s = match self {
            UnwrapperKind::Option => "unwrap_option_return_type",
            UnwrapperKind::Result => "unwrap_result_return_type",
        };

        AssistId::refactor_rewrite(s)
    }

    fn label(&self) -> &'static str {
        match self {
            UnwrapperKind::Option => "Unwrap Option return type",
            UnwrapperKind::Result => "Unwrap Result return type",
        }
    }

    fn core_type(&self, famous_defs: &FamousDefs<'_, '_>) -> Option<hir::Enum> {
        match self {
            UnwrapperKind::Option => famous_defs.core_option_Option(),
            UnwrapperKind::Result => famous_defs.core_result_Result(),
        }
    }
}

fn tail_cb_impl(acc: &mut Vec<ast::Expr>, e: &ast::Expr) {
    match e {
        ast::Expr::BreakExpr(break_expr) => {
            if let Some(break_expr_arg) = break_expr.expr() {
                for_each_tail_expr(&break_expr_arg, &mut |e| tail_cb_impl(acc, e))
            }
        }
        ast::Expr::ReturnExpr(_) => {
            // all return expressions have already been handled by the walk loop
        }
        e => acc.push(e.clone()),
    }
}

// Tries to extract `T` from `Option<T>` or `Result<T, E>`.
fn extract_wrapped_type(ty: &ast::Type) -> Option<ast::Type> {
    let ast::Type::PathType(path_ty) = ty else {
        return None;
    };
    let path = path_ty.path()?;
    let segment = path.first_segment()?;
    let generic_arg_list = segment.generic_arg_list()?;
    let generic_args: Vec<_> = generic_arg_list.generic_args().collect();
    let ast::GenericArg::TypeArg(happy_type) = generic_args.first()? else {
        return None;
    };
    happy_type.ty()
}

fn is_unit_type(ty: &ast::Type) -> bool {
    let ast::Type::TupleType(tuple) = ty else { return false };
    tuple.fields().next().is_none()
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_assist_by_label, check_assist_not_applicable_by_label};

    use super::*;

    #[test]
    fn unwrap_option_return_type_simple() {
        check_assist_by_label(
            unwrap_return_type,
            r#"
//- minicore: option
fn foo() -> Option<i3$02> {
    let test = "test";
    return Some(42i32);
}
"#,
            r#"
fn foo() -> i32 {
    let test = "test";
    return 42i32;
}
"#,
            "Unwrap Option return type",
        );
    }

    #[test]
    fn unwrap_option_return_type_unit_type() {
        check_assist_by_label(
            unwrap_return_type,
            r#"
//- minicore: option
fn foo() -> Option<()$0> {
    Some(())
}
"#,
            r#"
fn foo() {}
"#,
            "Unwrap Option return type",
        );

        // Unformatted return type
        check_assist_by_label(
            unwrap_return_type,
            r#"
//- minicore: option
fn foo() -> Option<()$0>{
    Some(())
}
"#,
            r#"
fn foo() {}
"#,
            "Unwrap Option return type",
        );
    }

    #[test]
    fn unwrap_option_return_type_none() {
        check_assist_by_label(
            unwrap_return_type,
            r#"
//- minicore: option
fn foo() -> Option<i3$02> {
    if true {
        Some(42)
    } else {
        None
    }
}
"#,
            r#"
fn foo() -> i32 {
    if true {
        42
    } else {
        ${1:()}$0
    }
}
"#,
            "Unwrap Option return type",
        );
    }

    #[test]
    fn unwrap_option_return_type_multi_none() {
        check_assist_by_label(
            unwrap_return_type,
            r#"
//- minicore: option
fn foo() -> Option<i3$02> {
    if false {
        return None;
    }

    if true {
        Some(42)
    } else {
        None
    }
}
"#,
            r#"
fn foo() -> i32 {
    if false {
        return ${1:()};
    }

    if true {
        42
    } else {
        ${2:()}$0
    }
}
"#,
            "Unwrap Option return type",
        );
    }

    #[test]
    fn unwrap_option_return_type_ending_with_parent() {
        check_assist_by_label(
            unwrap_return_type,
            r#"
//- minicore: option
fn foo() -> Option<i3$02> {
    if true {
        Some(42)
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
            "Unwrap Option return type",
        );
    }

    #[test]
    fn unwrap_option_return_type_break_split_tail() {
        check_assist_by_label(
            unwrap_return_type,
            r#"
//- minicore: option
fn foo() -> Option<i3$02> {
    loop {
        break if true {
            Some(1)
        } else {
            Some(0)
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
            "Unwrap Option return type",
        );
    }

    #[test]
    fn unwrap_option_return_type_simple_closure() {
        check_assist_by_label(
            unwrap_return_type,
            r#"
//- minicore: option
fn foo() {
    || -> Option<i32$0> {
        let test = "test";
        return Some(42i32);
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
            "Unwrap Option return type",
        );
    }

    #[test]
    fn unwrap_option_return_type_simple_return_type_bad_cursor() {
        check_assist_not_applicable_by_label(
            unwrap_return_type,
            r#"
//- minicore: option
fn foo() -> i32 {
    let test = "test";$0
    return 42i32;
}
"#,
            "Unwrap Option return type",
        );
    }

    #[test]
    fn unwrap_option_return_type_simple_return_type_bad_cursor_closure() {
        check_assist_not_applicable_by_label(
            unwrap_return_type,
            r#"
//- minicore: option
fn foo() {
    || -> i32 {
        let test = "test";$0
        return 42i32;
    };
}
"#,
            "Unwrap Option return type",
        );
    }

    #[test]
    fn unwrap_option_return_type_closure_non_block() {
        check_assist_not_applicable_by_label(
            unwrap_return_type,
            r#"
//- minicore: option
fn foo() { || -> i$032 3; }
"#,
            "Unwrap Option return type",
        );
    }

    #[test]
    fn unwrap_option_return_type_simple_return_type_already_not_option_std() {
        check_assist_not_applicable_by_label(
            unwrap_return_type,
            r#"
//- minicore: option
fn foo() -> i32$0 {
    let test = "test";
    return 42i32;
}
"#,
            "Unwrap Option return type",
        );
    }

    #[test]
    fn unwrap_option_return_type_simple_return_type_already_not_option_closure() {
        check_assist_not_applicable_by_label(
            unwrap_return_type,
            r#"
//- minicore: option
fn foo() {
    || -> i32$0 {
        let test = "test";
        return 42i32;
    };
}
"#,
            "Unwrap Option return type",
        );
    }

    #[test]
    fn unwrap_option_return_type_simple_with_tail() {
        check_assist_by_label(
            unwrap_return_type,
            r#"
//- minicore: option
fn foo() ->$0 Option<i32> {
    let test = "test";
    Some(42i32)
}
"#,
            r#"
fn foo() -> i32 {
    let test = "test";
    42i32
}
"#,
            "Unwrap Option return type",
        );
    }

    #[test]
    fn unwrap_option_return_type_simple_with_tail_closure() {
        check_assist_by_label(
            unwrap_return_type,
            r#"
//- minicore: option
fn foo() {
    || ->$0 Option<i32> {
        let test = "test";
        Some(42i32)
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
            "Unwrap Option return type",
        );
    }

    #[test]
    fn unwrap_option_return_type_simple_with_tail_only() {
        check_assist_by_label(
            unwrap_return_type,
            r#"
//- minicore: option
fn foo() -> Option<i32$0> { Some(42i32) }
"#,
            r#"
fn foo() -> i32 { 42i32 }
"#,
            "Unwrap Option return type",
        );
    }

    #[test]
    fn unwrap_option_return_type_simple_with_tail_block_like() {
        check_assist_by_label(
            unwrap_return_type,
            r#"
//- minicore: option
fn foo() -> Option<i32>$0 {
    if true {
        Some(42i32)
    } else {
        Some(24i32)
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
            "Unwrap Option return type",
        );
    }

    #[test]
    fn unwrap_option_return_type_simple_without_block_closure() {
        check_assist_by_label(
            unwrap_return_type,
            r#"
//- minicore: option
fn foo() {
    || -> Option<i32>$0 {
        if true {
            Some(42i32)
        } else {
            Some(24i32)
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
            "Unwrap Option return type",
        );
    }

    #[test]
    fn unwrap_option_return_type_simple_with_nested_if() {
        check_assist_by_label(
            unwrap_return_type,
            r#"
//- minicore: option
fn foo() -> Option<i32>$0 {
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
            "Unwrap Option return type",
        );
    }

    #[test]
    fn unwrap_option_return_type_simple_with_await() {
        check_assist_by_label(
            unwrap_return_type,
            r#"
//- minicore: option
async fn foo() -> Option<i$032> {
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
            "Unwrap Option return type",
        );
    }

    #[test]
    fn unwrap_option_return_type_simple_with_array() {
        check_assist_by_label(
            unwrap_return_type,
            r#"
//- minicore: option
fn foo() -> Option<[i32; 3]$0> { Some([1, 2, 3]) }
"#,
            r#"
fn foo() -> [i32; 3] { [1, 2, 3] }
"#,
            "Unwrap Option return type",
        );
    }

    #[test]
    fn unwrap_option_return_type_simple_with_cast() {
        check_assist_by_label(
            unwrap_return_type,
            r#"
//- minicore: option
fn foo() -$0> Option<i32> {
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
            "Unwrap Option return type",
        );
    }

    #[test]
    fn unwrap_option_return_type_simple_with_tail_block_like_match() {
        check_assist_by_label(
            unwrap_return_type,
            r#"
//- minicore: option
fn foo() -> Option<i32$0> {
    let my_var = 5;
    match my_var {
        5 => Some(42i32),
        _ => Some(24i32),
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
            "Unwrap Option return type",
        );
    }

    #[test]
    fn unwrap_option_return_type_simple_with_loop_with_tail() {
        check_assist_by_label(
            unwrap_return_type,
            r#"
//- minicore: option
fn foo() -> Option<i32$0> {
    let my_var = 5;
    loop {
        println!("test");
        5
    }
    Some(my_var)
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
            "Unwrap Option return type",
        );
    }

    #[test]
    fn unwrap_option_return_type_simple_with_loop_in_let_stmt() {
        check_assist_by_label(
            unwrap_return_type,
            r#"
//- minicore: option
fn foo() -> Option<i32$0> {
    let my_var = let x = loop {
        break 1;
    };
    Some(my_var)
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
            "Unwrap Option return type",
        );
    }

    #[test]
    fn unwrap_option_return_type_simple_with_tail_block_like_match_return_expr() {
        check_assist_by_label(
            unwrap_return_type,
            r#"
//- minicore: option
fn foo() -> Option<i32>$0 {
    let my_var = 5;
    let res = match my_var {
        5 => 42i32,
        _ => return Some(24i32),
    };
    Some(res)
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
            "Unwrap Option return type",
        );

        check_assist_by_label(
            unwrap_return_type,
            r#"
//- minicore: option
fn foo() -> Option<i32$0> {
    let my_var = 5;
    let res = if my_var == 5 {
        42i32
    } else {
        return Some(24i32);
    };
    Some(res)
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
            "Unwrap Option return type",
        );
    }

    #[test]
    fn unwrap_option_return_type_simple_with_tail_block_like_match_deeper() {
        check_assist_by_label(
            unwrap_return_type,
            r#"
//- minicore: option
fn foo() -> Option<i32$0> {
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
            "Unwrap Option return type",
        );
    }

    #[test]
    fn unwrap_option_return_type_simple_with_tail_block_like_early_return() {
        check_assist_by_label(
            unwrap_return_type,
            r#"
//- minicore: option
fn foo() -> Option<i32$0> {
    let test = "test";
    if test == "test" {
        return Some(24i32);
    }
    Some(53i32)
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
            "Unwrap Option return type",
        );
    }

    #[test]
    fn unwrap_option_return_in_tail_position() {
        check_assist_by_label(
            unwrap_return_type,
            r#"
//- minicore: option
fn foo(num: i32) -> $0Option<i32> {
    return Some(num)
}
"#,
            r#"
fn foo(num: i32) -> i32 {
    return num
}
"#,
            "Unwrap Option return type",
        );
    }

    #[test]
    fn unwrap_option_return_type_simple_with_closure() {
        check_assist_by_label(
            unwrap_return_type,
            r#"
//- minicore: option
fn foo(the_field: u32) -> Option<u32$0> {
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
            "Unwrap Option return type",
        );

        check_assist_by_label(
            unwrap_return_type,
            r#"
//- minicore: option
fn foo(the_field: u32) -> Option<u32$0> {
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
            "Unwrap Option return type",
        );
    }

    #[test]
    fn unwrap_option_return_type_simple_with_weird_forms() {
        check_assist_by_label(
            unwrap_return_type,
            r#"
//- minicore: option
fn foo() -> Option<i32$0> {
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
            "Unwrap Option return type",
        );

        check_assist_by_label(
            unwrap_return_type,
            r#"
//- minicore: option
fn foo(the_field: u32) -> Option<u32$0> {
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
            "Unwrap Option return type",
        );

        check_assist_by_label(
            unwrap_return_type,
            r#"
//- minicore: option
fn foo(the_field: u32) -> Option<u32$0> {
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
            "Unwrap Option return type",
        );

        check_assist_by_label(
            unwrap_return_type,
            r#"
//- minicore: option
fn foo(the_field: u32) -> Option<u32$0> {
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
            "Unwrap Option return type",
        );

        check_assist_by_label(
            unwrap_return_type,
            r#"
//- minicore: option
fn foo(the_field: u32) -> Option<u3$02> {
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
            "Unwrap Option return type",
        );
    }

    #[test]
    fn unwrap_option_return_type_nested_type() {
        check_assist_by_label(
            unwrap_return_type,
            r#"
//- minicore: option, result
fn foo() -> Option<Result<i32$0, ()>> {
    Some(Ok(42))
}
"#,
            r#"
fn foo() -> Result<i32, ()> {
    Ok(42)
}
"#,
            "Unwrap Option return type",
        );

        check_assist_by_label(
            unwrap_return_type,
            r#"
//- minicore: option, result
fn foo() -> Option<Result<Option<i32$0>, ()>> {
    Some(Err())
}
"#,
            r#"
fn foo() -> Result<Option<i32>, ()> {
    Err()
}
"#,
            "Unwrap Option return type",
        );

        check_assist_by_label(
            unwrap_return_type,
            r#"
//- minicore: option, result, iterators
fn foo() -> Option<impl Iterator<Item = i32>$0> {
    Some(Some(42).into_iter())
}
"#,
            r#"
fn foo() -> impl Iterator<Item = i32> {
    Some(42).into_iter()
}
"#,
            "Unwrap Option return type",
        );
    }

    #[test]
    fn unwrap_result_return_type_simple() {
        check_assist_by_label(
            unwrap_return_type,
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
            "Unwrap Result return type",
        );
    }

    #[test]
    fn unwrap_result_return_type_unit_type() {
        check_assist_by_label(
            unwrap_return_type,
            r#"
//- minicore: result
fn foo() -> Result<(), Box<dyn Error$0>> {
    Ok(())
}
"#,
            r#"
fn foo() {}
"#,
            "Unwrap Result return type",
        );

        // Unformatted return type
        check_assist_by_label(
            unwrap_return_type,
            r#"
//- minicore: result
fn foo() -> Result<(), Box<dyn Error$0>>{
    Ok(())
}
"#,
            r#"
fn foo() {}
"#,
            "Unwrap Result return type",
        );
    }

    #[test]
    fn unwrap_result_return_type_ending_with_parent() {
        check_assist_by_label(
            unwrap_return_type,
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
            "Unwrap Result return type",
        );
    }

    #[test]
    fn unwrap_result_return_type_break_split_tail() {
        check_assist_by_label(
            unwrap_return_type,
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
            "Unwrap Result return type",
        );
    }

    #[test]
    fn unwrap_result_return_type_simple_closure() {
        check_assist_by_label(
            unwrap_return_type,
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
            "Unwrap Result return type",
        );
    }

    #[test]
    fn unwrap_result_return_type_simple_return_type_bad_cursor() {
        check_assist_not_applicable_by_label(
            unwrap_return_type,
            r#"
//- minicore: result
fn foo() -> i32 {
    let test = "test";$0
    return 42i32;
}
"#,
            "Unwrap Result return type",
        );
    }

    #[test]
    fn unwrap_result_return_type_simple_return_type_bad_cursor_closure() {
        check_assist_not_applicable_by_label(
            unwrap_return_type,
            r#"
//- minicore: result
fn foo() {
    || -> i32 {
        let test = "test";$0
        return 42i32;
    };
}
"#,
            "Unwrap Result return type",
        );
    }

    #[test]
    fn unwrap_result_return_type_closure_non_block() {
        check_assist_not_applicable_by_label(
            unwrap_return_type,
            r#"
//- minicore: result
fn foo() { || -> i$032 3; }
"#,
            "Unwrap Result return type",
        );
    }

    #[test]
    fn unwrap_result_return_type_simple_return_type_already_not_result_std() {
        check_assist_not_applicable_by_label(
            unwrap_return_type,
            r#"
//- minicore: result
fn foo() -> i32$0 {
    let test = "test";
    return 42i32;
}
"#,
            "Unwrap Result return type",
        );
    }

    #[test]
    fn unwrap_result_return_type_simple_return_type_already_not_result_closure() {
        check_assist_not_applicable_by_label(
            unwrap_return_type,
            r#"
//- minicore: result
fn foo() {
    || -> i32$0 {
        let test = "test";
        return 42i32;
    };
}
"#,
            "Unwrap Result return type",
        );
    }

    #[test]
    fn unwrap_result_return_type_simple_with_tail() {
        check_assist_by_label(
            unwrap_return_type,
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
            "Unwrap Result return type",
        );
    }

    #[test]
    fn unwrap_result_return_type_simple_with_tail_closure() {
        check_assist_by_label(
            unwrap_return_type,
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
            "Unwrap Result return type",
        );
    }

    #[test]
    fn unwrap_result_return_type_simple_with_tail_only() {
        check_assist_by_label(
            unwrap_return_type,
            r#"
//- minicore: result
fn foo() -> Result<i32$0> { Ok(42i32) }
"#,
            r#"
fn foo() -> i32 { 42i32 }
"#,
            "Unwrap Result return type",
        );
    }

    #[test]
    fn unwrap_result_return_type_simple_with_tail_block_like() {
        check_assist_by_label(
            unwrap_return_type,
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
            "Unwrap Result return type",
        );
    }

    #[test]
    fn unwrap_result_return_type_simple_without_block_closure() {
        check_assist_by_label(
            unwrap_return_type,
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
            "Unwrap Result return type",
        );
    }

    #[test]
    fn unwrap_result_return_type_simple_with_nested_if() {
        check_assist_by_label(
            unwrap_return_type,
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
            "Unwrap Result return type",
        );
    }

    #[test]
    fn unwrap_result_return_type_simple_with_await() {
        check_assist_by_label(
            unwrap_return_type,
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
            "Unwrap Result return type",
        );
    }

    #[test]
    fn unwrap_result_return_type_simple_with_array() {
        check_assist_by_label(
            unwrap_return_type,
            r#"
//- minicore: result
fn foo() -> Result<[i32; 3]$0> { Ok([1, 2, 3]) }
"#,
            r#"
fn foo() -> [i32; 3] { [1, 2, 3] }
"#,
            "Unwrap Result return type",
        );
    }

    #[test]
    fn unwrap_result_return_type_simple_with_cast() {
        check_assist_by_label(
            unwrap_return_type,
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
            "Unwrap Result return type",
        );
    }

    #[test]
    fn unwrap_result_return_type_simple_with_tail_block_like_match() {
        check_assist_by_label(
            unwrap_return_type,
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
            "Unwrap Result return type",
        );
    }

    #[test]
    fn unwrap_result_return_type_simple_with_loop_with_tail() {
        check_assist_by_label(
            unwrap_return_type,
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
            "Unwrap Result return type",
        );
    }

    #[test]
    fn unwrap_result_return_type_simple_with_loop_in_let_stmt() {
        check_assist_by_label(
            unwrap_return_type,
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
            "Unwrap Result return type",
        );
    }

    #[test]
    fn unwrap_result_return_type_simple_with_tail_block_like_match_return_expr() {
        check_assist_by_label(
            unwrap_return_type,
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
            "Unwrap Result return type",
        );

        check_assist_by_label(
            unwrap_return_type,
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
            "Unwrap Result return type",
        );
    }

    #[test]
    fn unwrap_result_return_type_simple_with_tail_block_like_match_deeper() {
        check_assist_by_label(
            unwrap_return_type,
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
            "Unwrap Result return type",
        );
    }

    #[test]
    fn unwrap_result_return_type_simple_with_tail_block_like_early_return() {
        check_assist_by_label(
            unwrap_return_type,
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
            "Unwrap Result return type",
        );
    }

    #[test]
    fn unwrap_result_return_in_tail_position() {
        check_assist_by_label(
            unwrap_return_type,
            r#"
//- minicore: result
fn foo(num: i32) -> $0Result<i32, String> {
    return Ok(num)
}
"#,
            r#"
fn foo(num: i32) -> i32 {
    return num
}
"#,
            "Unwrap Result return type",
        );
    }

    #[test]
    fn unwrap_result_return_type_simple_with_closure() {
        check_assist_by_label(
            unwrap_return_type,
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
            "Unwrap Result return type",
        );

        check_assist_by_label(
            unwrap_return_type,
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
            "Unwrap Result return type",
        );
    }

    #[test]
    fn unwrap_result_return_type_simple_with_weird_forms() {
        check_assist_by_label(
            unwrap_return_type,
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
            "Unwrap Result return type",
        );

        check_assist_by_label(
            unwrap_return_type,
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
            "Unwrap Result return type",
        );

        check_assist_by_label(
            unwrap_return_type,
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
            "Unwrap Result return type",
        );

        check_assist_by_label(
            unwrap_return_type,
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
            "Unwrap Result return type",
        );

        check_assist_by_label(
            unwrap_return_type,
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
            "Unwrap Result return type",
        );
    }

    #[test]
    fn unwrap_result_return_type_nested_type() {
        check_assist_by_label(
            unwrap_return_type,
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
            "Unwrap Result return type",
        );

        check_assist_by_label(
            unwrap_return_type,
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
            "Unwrap Result return type",
        );

        check_assist_by_label(
            unwrap_return_type,
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
            "Unwrap Result return type",
        );
    }
}
