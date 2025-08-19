use std::iter;

use hir::HasSource;
use ide_db::{
    assists::GroupLabel,
    famous_defs::FamousDefs,
    syntax_helpers::node_ext::{for_each_tail_expr, walk_expr},
};
use syntax::{
    AstNode,
    ast::{self, Expr, HasGenericArgs, HasGenericParams, syntax_factory::SyntaxFactory},
    match_ast,
};

use crate::{AssistContext, AssistId, Assists};

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

pub(crate) fn wrap_return_type(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    let ret_type = ctx.find_node_at_offset::<ast::RetType>()?;
    let parent = ret_type.syntax().parent()?;
    let body_expr = match_ast! {
        match parent {
            ast::Fn(func) => func.body()?.into(),
            ast::ClosureExpr(closure) => match closure.body()? {
                Expr::BlockExpr(block) => block.into(),
                // closures require a block when a return type is specified
                _ => return None,
            },
            _ => return None,
        }
    };

    let type_ref = &ret_type.ty()?;
    let ty = ctx.sema.resolve_type(type_ref)?;
    let ty_adt = ty.as_adt();
    let famous_defs = FamousDefs(&ctx.sema, ctx.sema.scope(type_ref.syntax())?.krate());

    for kind in WrapperKind::ALL {
        let Some(core_wrapper) = kind.core_type(&famous_defs) else {
            continue;
        };

        if matches!(ty_adt, Some(hir::Adt::Enum(ret_type)) if ret_type == core_wrapper) {
            // The return type is already wrapped
            cov_mark::hit!(wrap_return_type_simple_return_type_already_wrapped);
            continue;
        }

        acc.add_group(
            &GroupLabel("Wrap return type in...".into()),
            kind.assist_id(),
            kind.label(),
            type_ref.syntax().text_range(),
            |builder| {
                let mut editor = builder.make_editor(&parent);
                let make = SyntaxFactory::with_mappings();
                let alias = wrapper_alias(ctx, &make, core_wrapper, type_ref, &ty, kind.symbol());
                let (ast_new_return_ty, semantic_new_return_ty) = alias.unwrap_or_else(|| {
                    let (ast_ty, ty_constructor) = match kind {
                        WrapperKind::Option => {
                            (make.ty_option(type_ref.clone()), famous_defs.core_option_Option())
                        }
                        WrapperKind::Result => (
                            make.ty_result(type_ref.clone(), make.ty_infer().into()),
                            famous_defs.core_result_Result(),
                        ),
                    };
                    let semantic_ty = ty_constructor
                        .map(|ty_constructor| {
                            hir::Adt::from(ty_constructor).ty_with_args(ctx.db(), [ty.clone()])
                        })
                        .unwrap_or_else(|| ty.clone());
                    (ast_ty, semantic_ty)
                });

                let mut exprs_to_wrap = Vec::new();
                let tail_cb = &mut |e: &_| tail_cb_impl(&mut exprs_to_wrap, e);
                walk_expr(&body_expr, &mut |expr| {
                    if let Expr::ReturnExpr(ret_expr) = expr
                        && let Some(ret_expr_arg) = &ret_expr.expr()
                    {
                        for_each_tail_expr(ret_expr_arg, tail_cb);
                    }
                });
                for_each_tail_expr(&body_expr, tail_cb);

                for ret_expr_arg in exprs_to_wrap {
                    if let Some(ty) = ctx.sema.type_of_expr(&ret_expr_arg)
                        && ty.adjusted().could_unify_with(ctx.db(), &semantic_new_return_ty)
                    {
                        // The type is already correct, don't wrap it.
                        // We deliberately don't use `could_unify_with_deeply()`, because as long as the outer
                        // enum matches it's okay for us, as we don't trigger the assist if the return type
                        // is already `Option`/`Result`, so mismatched exact type is more likely a mistake
                        // than something intended.
                        continue;
                    }

                    let happy_wrapped = make.expr_call(
                        make.expr_path(make.ident_path(kind.happy_ident())),
                        make.arg_list(iter::once(ret_expr_arg.clone())),
                    );
                    editor.replace(ret_expr_arg.syntax(), happy_wrapped.syntax());
                }

                editor.replace(type_ref.syntax(), ast_new_return_ty.syntax());

                if let WrapperKind::Result = kind {
                    // Add a placeholder snippet at the first generic argument that doesn't equal the return type.
                    // This is normally the error type, but that may not be the case when we inserted a type alias.
                    let args = ast_new_return_ty
                        .path()
                        .unwrap()
                        .segment()
                        .unwrap()
                        .generic_arg_list()
                        .unwrap();
                    let error_type_arg = args.generic_args().find(|arg| match arg {
                        ast::GenericArg::TypeArg(_) => {
                            arg.syntax().text() != type_ref.syntax().text()
                        }
                        ast::GenericArg::LifetimeArg(_) => false,
                        _ => true,
                    });
                    if let Some(error_type_arg) = error_type_arg
                        && let Some(cap) = ctx.config.snippet_cap
                    {
                        editor.add_annotation(
                            error_type_arg.syntax(),
                            builder.make_placeholder_snippet(cap),
                        );
                    }
                }

                editor.add_mappings(make.finish_with_mappings());
                builder.add_file_edits(ctx.vfs_file_id(), editor);
            },
        );
    }

    Some(())
}

enum WrapperKind {
    Option,
    Result,
}

impl WrapperKind {
    const ALL: &'static [WrapperKind] = &[WrapperKind::Option, WrapperKind::Result];

    fn assist_id(&self) -> AssistId {
        let s = match self {
            WrapperKind::Option => "wrap_return_type_in_option",
            WrapperKind::Result => "wrap_return_type_in_result",
        };

        AssistId::refactor_rewrite(s)
    }

    fn label(&self) -> &'static str {
        match self {
            WrapperKind::Option => "Wrap return type in Option",
            WrapperKind::Result => "Wrap return type in Result",
        }
    }

    fn happy_ident(&self) -> &'static str {
        match self {
            WrapperKind::Option => "Some",
            WrapperKind::Result => "Ok",
        }
    }

    fn core_type(&self, famous_defs: &FamousDefs<'_, '_>) -> Option<hir::Enum> {
        match self {
            WrapperKind::Option => famous_defs.core_option_Option(),
            WrapperKind::Result => famous_defs.core_result_Result(),
        }
    }

    fn symbol(&self) -> hir::Symbol {
        match self {
            WrapperKind::Option => hir::sym::Option,
            WrapperKind::Result => hir::sym::Result,
        }
    }
}

// Try to find an wrapper type alias in the current scope (shadowing the default).
fn wrapper_alias<'db>(
    ctx: &AssistContext<'db>,
    make: &SyntaxFactory,
    core_wrapper: hir::Enum,
    ast_ret_type: &ast::Type,
    semantic_ret_type: &hir::Type<'db>,
    wrapper: hir::Symbol,
) -> Option<(ast::PathType, hir::Type<'db>)> {
    let wrapper_path = hir::ModPath::from_segments(
        hir::PathKind::Plain,
        iter::once(hir::Name::new_symbol_root(wrapper)),
    );

    ctx.sema.resolve_mod_path(ast_ret_type.syntax(), &wrapper_path).and_then(|def| {
        def.filter_map(|def| match def.into_module_def() {
            hir::ModuleDef::TypeAlias(alias) => {
                let enum_ty = alias.ty(ctx.db()).as_adt()?.as_enum()?;
                (enum_ty == core_wrapper).then_some((alias, enum_ty))
            }
            _ => None,
        })
        .find_map(|(alias, enum_ty)| {
            let mut inserted_ret_type = false;
            let generic_args =
                alias.source(ctx.db())?.value.generic_param_list()?.generic_params().map(|param| {
                    match param {
                        // Replace the very first type parameter with the function's return type.
                        ast::GenericParam::TypeParam(_) if !inserted_ret_type => {
                            inserted_ret_type = true;
                            make.type_arg(ast_ret_type.clone()).into()
                        }
                        ast::GenericParam::LifetimeParam(_) => {
                            make.lifetime_arg(make.lifetime("'_")).into()
                        }
                        _ => make.type_arg(make.ty_infer().into()).into(),
                    }
                });

            let name = alias.name(ctx.db());
            let generic_arg_list = make.generic_arg_list(generic_args, false);
            let path = make.path_unqualified(
                make.path_segment_generics(make.name_ref(name.as_str()), generic_arg_list),
            );

            let new_ty =
                hir::Adt::from(enum_ty).ty_with_args(ctx.db(), [semantic_ret_type.clone()]);

            Some((make.ty_path(path), new_ty))
        })
    })
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
    use crate::tests::{check_assist_by_label, check_assist_not_applicable_by_label};

    use super::*;

    #[test]
    fn wrap_return_type_in_option_simple() {
        check_assist_by_label(
            wrap_return_type,
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
            WrapperKind::Option.label(),
        );
    }

    #[test]
    fn wrap_return_type_in_option_break_split_tail() {
        check_assist_by_label(
            wrap_return_type,
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
            WrapperKind::Option.label(),
        );
    }

    #[test]
    fn wrap_return_type_in_option_simple_closure() {
        check_assist_by_label(
            wrap_return_type,
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
            WrapperKind::Option.label(),
        );
    }

    #[test]
    fn wrap_return_type_in_option_simple_return_type_bad_cursor() {
        check_assist_not_applicable_by_label(
            wrap_return_type,
            r#"
//- minicore: option
fn foo() -> i32 {
    let test = "test";$0
    return 42i32;
}
"#,
            WrapperKind::Option.label(),
        );
    }

    #[test]
    fn wrap_return_type_in_option_simple_return_type_bad_cursor_closure() {
        check_assist_not_applicable_by_label(
            wrap_return_type,
            r#"
//- minicore: option
fn foo() {
    || -> i32 {
        let test = "test";$0
        return 42i32;
    };
}
"#,
            WrapperKind::Option.label(),
        );
    }

    #[test]
    fn wrap_return_type_in_option_closure_non_block() {
        check_assist_not_applicable_by_label(
            wrap_return_type,
            r#"
//- minicore: option
fn foo() { || -> i$032 3; }
"#,
            WrapperKind::Option.label(),
        );
    }

    #[test]
    fn wrap_return_type_in_option_simple_return_type_already_option_std() {
        check_assist_not_applicable_by_label(
            wrap_return_type,
            r#"
//- minicore: option
fn foo() -> core::option::Option<i32$0> {
    let test = "test";
    return 42i32;
}
"#,
            WrapperKind::Option.label(),
        );
    }

    #[test]
    fn wrap_return_type_in_option_simple_return_type_already_option() {
        cov_mark::check!(wrap_return_type_simple_return_type_already_wrapped);
        check_assist_not_applicable_by_label(
            wrap_return_type,
            r#"
//- minicore: option
fn foo() -> Option<i32$0> {
    let test = "test";
    return 42i32;
}
"#,
            WrapperKind::Option.label(),
        );
    }

    #[test]
    fn wrap_return_type_in_option_simple_return_type_already_option_closure() {
        check_assist_not_applicable_by_label(
            wrap_return_type,
            r#"
//- minicore: option
fn foo() {
    || -> Option<i32$0, String> {
        let test = "test";
        return 42i32;
    };
}
"#,
            WrapperKind::Option.label(),
        );
    }

    #[test]
    fn wrap_return_type_in_option_simple_with_cursor() {
        check_assist_by_label(
            wrap_return_type,
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
            WrapperKind::Option.label(),
        );
    }

    #[test]
    fn wrap_return_type_in_option_simple_with_tail() {
        check_assist_by_label(
            wrap_return_type,
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
            WrapperKind::Option.label(),
        );
    }

    #[test]
    fn wrap_return_type_in_option_simple_with_tail_closure() {
        check_assist_by_label(
            wrap_return_type,
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
            WrapperKind::Option.label(),
        );
    }

    #[test]
    fn wrap_return_type_in_option_simple_with_tail_only() {
        check_assist_by_label(
            wrap_return_type,
            r#"
//- minicore: option
fn foo() -> i32$0 { 42i32 }
"#,
            r#"
fn foo() -> Option<i32> { Some(42i32) }
"#,
            WrapperKind::Option.label(),
        );
    }

    #[test]
    fn wrap_return_type_in_option_simple_with_tail_block_like() {
        check_assist_by_label(
            wrap_return_type,
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
            WrapperKind::Option.label(),
        );
    }

    #[test]
    fn wrap_return_type_in_option_simple_without_block_closure() {
        check_assist_by_label(
            wrap_return_type,
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
            WrapperKind::Option.label(),
        );
    }

    #[test]
    fn wrap_return_type_in_option_simple_with_nested_if() {
        check_assist_by_label(
            wrap_return_type,
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
            WrapperKind::Option.label(),
        );
    }

    #[test]
    fn wrap_return_type_in_option_simple_with_await() {
        check_assist_by_label(
            wrap_return_type,
            r#"
//- minicore: option, future
struct F(i32);
impl core::future::Future for F {
    type Output = i32;
    fn poll(self: core::pin::Pin<&mut Self>, cx: &mut core::task::Context<'_>) -> core::task::Poll<Self::Output> { 0 }
}
async fn foo() -> i$032 {
    if true {
        if false {
            F(1).await
        } else {
            F(2).await
        }
    } else {
        F(24i32).await
    }
}
"#,
            r#"
struct F(i32);
impl core::future::Future for F {
    type Output = i32;
    fn poll(self: core::pin::Pin<&mut Self>, cx: &mut core::task::Context<'_>) -> core::task::Poll<Self::Output> { 0 }
}
async fn foo() -> Option<i32> {
    if true {
        if false {
            Some(F(1).await)
        } else {
            Some(F(2).await)
        }
    } else {
        Some(F(24i32).await)
    }
}
"#,
            WrapperKind::Option.label(),
        );
    }

    #[test]
    fn wrap_return_type_in_option_simple_with_array() {
        check_assist_by_label(
            wrap_return_type,
            r#"
//- minicore: option
fn foo() -> [i32;$0 3] { [1, 2, 3] }
"#,
            r#"
fn foo() -> Option<[i32; 3]> { Some([1, 2, 3]) }
"#,
            WrapperKind::Option.label(),
        );
    }

    #[test]
    fn wrap_return_type_in_option_simple_with_cast() {
        check_assist_by_label(
            wrap_return_type,
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
            WrapperKind::Option.label(),
        );
    }

    #[test]
    fn wrap_return_type_in_option_simple_with_tail_block_like_match() {
        check_assist_by_label(
            wrap_return_type,
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
            WrapperKind::Option.label(),
        );
    }

    #[test]
    fn wrap_return_type_in_option_simple_with_loop_with_tail() {
        check_assist_by_label(
            wrap_return_type,
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
            WrapperKind::Option.label(),
        );
    }

    #[test]
    fn wrap_return_type_in_option_simple_with_loop_in_let_stmt() {
        check_assist_by_label(
            wrap_return_type,
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
            WrapperKind::Option.label(),
        );
    }

    #[test]
    fn wrap_return_type_in_option_simple_with_tail_block_like_match_return_expr() {
        check_assist_by_label(
            wrap_return_type,
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
            WrapperKind::Option.label(),
        );

        check_assist_by_label(
            wrap_return_type,
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
            WrapperKind::Option.label(),
        );
    }

    #[test]
    fn wrap_return_type_in_option_simple_with_tail_block_like_match_deeper() {
        check_assist_by_label(
            wrap_return_type,
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
            WrapperKind::Option.label(),
        );
    }

    #[test]
    fn wrap_return_type_in_option_simple_with_tail_block_like_early_return() {
        check_assist_by_label(
            wrap_return_type,
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
            WrapperKind::Option.label(),
        );
    }

    #[test]
    fn wrap_return_in_option_tail_position() {
        check_assist_by_label(
            wrap_return_type,
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
            WrapperKind::Option.label(),
        );
    }

    #[test]
    fn wrap_return_type_in_option_simple_with_closure() {
        check_assist_by_label(
            wrap_return_type,
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
            WrapperKind::Option.label(),
        );

        check_assist_by_label(
            wrap_return_type,
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
            WrapperKind::Option.label(),
        );
    }

    #[test]
    fn wrap_return_type_in_option_simple_with_weird_forms() {
        check_assist_by_label(
            wrap_return_type,
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
            WrapperKind::Option.label(),
        );

        check_assist_by_label(
            wrap_return_type,
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
            WrapperKind::Option.label(),
        );

        check_assist_by_label(
            wrap_return_type,
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
            WrapperKind::Option.label(),
        );

        check_assist_by_label(
            wrap_return_type,
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
            WrapperKind::Option.label(),
        );

        check_assist_by_label(
            wrap_return_type,
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
            WrapperKind::Option.label(),
        );
    }

    #[test]
    fn wrap_return_type_in_local_option_type() {
        check_assist_by_label(
            wrap_return_type,
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
            WrapperKind::Option.label(),
        );

        check_assist_by_label(
            wrap_return_type,
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
            WrapperKind::Option.label(),
        );
    }

    #[test]
    fn wrap_return_type_in_imported_local_option_type() {
        check_assist_by_label(
            wrap_return_type,
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
            WrapperKind::Option.label(),
        );

        check_assist_by_label(
            wrap_return_type,
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
            WrapperKind::Option.label(),
        );
    }

    #[test]
    fn wrap_return_type_in_local_option_type_from_function_body() {
        check_assist_by_label(
            wrap_return_type,
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
            WrapperKind::Option.label(),
        );
    }

    #[test]
    fn wrap_return_type_in_local_option_type_already_using_alias() {
        check_assist_not_applicable_by_label(
            wrap_return_type,
            r#"
//- minicore: option
pub type Option<T> = core::option::Option<T>;

fn foo() -> Option<i3$02> {
    return Some(42i32);
}
"#,
            WrapperKind::Option.label(),
        );
    }

    #[test]
    fn wrap_return_type_in_result_simple() {
        check_assist_by_label(
            wrap_return_type,
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
            WrapperKind::Result.label(),
        );
    }

    #[test]
    fn wrap_return_type_in_result_break_split_tail() {
        check_assist_by_label(
            wrap_return_type,
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
            WrapperKind::Result.label(),
        );
    }

    #[test]
    fn wrap_return_type_in_result_simple_closure() {
        check_assist_by_label(
            wrap_return_type,
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
            WrapperKind::Result.label(),
        );
    }

    #[test]
    fn wrap_return_type_in_result_simple_return_type_bad_cursor() {
        check_assist_not_applicable_by_label(
            wrap_return_type,
            r#"
//- minicore: result
fn foo() -> i32 {
    let test = "test";$0
    return 42i32;
}
"#,
            WrapperKind::Result.label(),
        );
    }

    #[test]
    fn wrap_return_type_in_result_simple_return_type_bad_cursor_closure() {
        check_assist_not_applicable_by_label(
            wrap_return_type,
            r#"
//- minicore: result
fn foo() {
    || -> i32 {
        let test = "test";$0
        return 42i32;
    };
}
"#,
            WrapperKind::Result.label(),
        );
    }

    #[test]
    fn wrap_return_type_in_result_closure_non_block() {
        check_assist_not_applicable_by_label(
            wrap_return_type,
            r#"
//- minicore: result
fn foo() { || -> i$032 3; }
"#,
            WrapperKind::Result.label(),
        );
    }

    #[test]
    fn wrap_return_type_in_result_simple_return_type_already_result_std() {
        check_assist_not_applicable_by_label(
            wrap_return_type,
            r#"
//- minicore: result
fn foo() -> core::result::Result<i32$0, String> {
    let test = "test";
    return 42i32;
}
"#,
            WrapperKind::Result.label(),
        );
    }

    #[test]
    fn wrap_return_type_in_result_simple_return_type_already_result() {
        cov_mark::check!(wrap_return_type_simple_return_type_already_wrapped);
        check_assist_not_applicable_by_label(
            wrap_return_type,
            r#"
//- minicore: result
fn foo() -> Result<i32$0, String> {
    let test = "test";
    return 42i32;
}
"#,
            WrapperKind::Result.label(),
        );
    }

    #[test]
    fn wrap_return_type_in_result_simple_return_type_already_result_closure() {
        check_assist_not_applicable_by_label(
            wrap_return_type,
            r#"
//- minicore: result
fn foo() {
    || -> Result<i32$0, String> {
        let test = "test";
        return 42i32;
    };
}
"#,
            WrapperKind::Result.label(),
        );
    }

    #[test]
    fn wrap_return_type_in_result_simple_with_cursor() {
        check_assist_by_label(
            wrap_return_type,
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
            WrapperKind::Result.label(),
        );
    }

    #[test]
    fn wrap_return_type_in_result_simple_with_tail() {
        check_assist_by_label(
            wrap_return_type,
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
            WrapperKind::Result.label(),
        );
    }

    #[test]
    fn wrap_return_type_in_result_simple_with_tail_closure() {
        check_assist_by_label(
            wrap_return_type,
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
            WrapperKind::Result.label(),
        );
    }

    #[test]
    fn wrap_return_type_in_result_simple_with_tail_only() {
        check_assist_by_label(
            wrap_return_type,
            r#"
//- minicore: result
fn foo() -> i32$0 { 42i32 }
"#,
            r#"
fn foo() -> Result<i32, ${0:_}> { Ok(42i32) }
"#,
            WrapperKind::Result.label(),
        );
    }

    #[test]
    fn wrap_return_type_in_result_simple_with_tail_block_like() {
        check_assist_by_label(
            wrap_return_type,
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
            WrapperKind::Result.label(),
        );
    }

    #[test]
    fn wrap_return_type_in_result_simple_without_block_closure() {
        check_assist_by_label(
            wrap_return_type,
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
            WrapperKind::Result.label(),
        );
    }

    #[test]
    fn wrap_return_type_in_result_simple_with_nested_if() {
        check_assist_by_label(
            wrap_return_type,
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
            WrapperKind::Result.label(),
        );
    }

    #[test]
    fn wrap_return_type_in_result_simple_with_await() {
        check_assist_by_label(
            wrap_return_type,
            r#"
//- minicore: result, future
struct F(i32);
impl core::future::Future for F {
    type Output = i32;
    fn poll(self: core::pin::Pin<&mut Self>, cx: &mut core::task::Context<'_>) -> core::task::Poll<Self::Output> { 0 }
}
async fn foo() -> i$032 {
    if true {
        if false {
            F(1).await
        } else {
            F(2).await
        }
    } else {
        F(24i32).await
    }
}
"#,
            r#"
struct F(i32);
impl core::future::Future for F {
    type Output = i32;
    fn poll(self: core::pin::Pin<&mut Self>, cx: &mut core::task::Context<'_>) -> core::task::Poll<Self::Output> { 0 }
}
async fn foo() -> Result<i32, ${0:_}> {
    if true {
        if false {
            Ok(F(1).await)
        } else {
            Ok(F(2).await)
        }
    } else {
        Ok(F(24i32).await)
    }
}
"#,
            WrapperKind::Result.label(),
        );
    }

    #[test]
    fn wrap_return_type_in_result_simple_with_array() {
        check_assist_by_label(
            wrap_return_type,
            r#"
//- minicore: result
fn foo() -> [i32;$0 3] { [1, 2, 3] }
"#,
            r#"
fn foo() -> Result<[i32; 3], ${0:_}> { Ok([1, 2, 3]) }
"#,
            WrapperKind::Result.label(),
        );
    }

    #[test]
    fn wrap_return_type_in_result_simple_with_cast() {
        check_assist_by_label(
            wrap_return_type,
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
            WrapperKind::Result.label(),
        );
    }

    #[test]
    fn wrap_return_type_in_result_simple_with_tail_block_like_match() {
        check_assist_by_label(
            wrap_return_type,
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
            WrapperKind::Result.label(),
        );
    }

    #[test]
    fn wrap_return_type_in_result_simple_with_loop_with_tail() {
        check_assist_by_label(
            wrap_return_type,
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
            WrapperKind::Result.label(),
        );
    }

    #[test]
    fn wrap_return_type_in_result_simple_with_loop_in_let_stmt() {
        check_assist_by_label(
            wrap_return_type,
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
            WrapperKind::Result.label(),
        );
    }

    #[test]
    fn wrap_return_type_in_result_simple_with_tail_block_like_match_return_expr() {
        check_assist_by_label(
            wrap_return_type,
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
            WrapperKind::Result.label(),
        );

        check_assist_by_label(
            wrap_return_type,
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
            WrapperKind::Result.label(),
        );
    }

    #[test]
    fn wrap_return_type_in_result_simple_with_tail_block_like_match_deeper() {
        check_assist_by_label(
            wrap_return_type,
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
            WrapperKind::Result.label(),
        );
    }

    #[test]
    fn wrap_return_type_in_result_simple_with_tail_block_like_early_return() {
        check_assist_by_label(
            wrap_return_type,
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
            WrapperKind::Result.label(),
        );
    }

    #[test]
    fn wrap_return_in_result_tail_position() {
        check_assist_by_label(
            wrap_return_type,
            r#"
//- minicore: result
fn foo(num: i32) -> $0i32 {
    return num
}
"#,
            r#"
fn foo(num: i32) -> Result<i32, ${0:_}> {
    return Ok(num)
}
"#,
            WrapperKind::Result.label(),
        );
    }

    #[test]
    fn wrap_return_type_in_result_simple_with_closure() {
        check_assist_by_label(
            wrap_return_type,
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
            WrapperKind::Result.label(),
        );

        check_assist_by_label(
            wrap_return_type,
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
            WrapperKind::Result.label(),
        );
    }

    #[test]
    fn wrap_return_type_in_result_simple_with_weird_forms() {
        check_assist_by_label(
            wrap_return_type,
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
            WrapperKind::Result.label(),
        );

        check_assist_by_label(
            wrap_return_type,
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
            WrapperKind::Result.label(),
        );

        check_assist_by_label(
            wrap_return_type,
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
            WrapperKind::Result.label(),
        );

        check_assist_by_label(
            wrap_return_type,
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
            WrapperKind::Result.label(),
        );

        check_assist_by_label(
            wrap_return_type,
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
            WrapperKind::Result.label(),
        );
    }

    #[test]
    fn wrap_return_type_in_local_result_type() {
        check_assist_by_label(
            wrap_return_type,
            r#"
//- minicore: result
type Result<T> = core::result::Result<T, ()>;

fn foo() -> i3$02 {
    return 42i32;
}
"#,
            r#"
type Result<T> = core::result::Result<T, ()>;

fn foo() -> Result<i32> {
    return Ok(42i32);
}
"#,
            WrapperKind::Result.label(),
        );

        check_assist_by_label(
            wrap_return_type,
            r#"
//- minicore: result
type Result2<T> = core::result::Result<T, ()>;

fn foo() -> i3$02 {
    return 42i32;
}
"#,
            r#"
type Result2<T> = core::result::Result<T, ()>;

fn foo() -> Result<i32, ${0:_}> {
    return Ok(42i32);
}
"#,
            WrapperKind::Result.label(),
        );
    }

    #[test]
    fn wrap_return_type_in_imported_local_result_type() {
        check_assist_by_label(
            wrap_return_type,
            r#"
//- minicore: result
mod some_module {
    pub type Result<T> = core::result::Result<T, ()>;
}

use some_module::Result;

fn foo() -> i3$02 {
    return 42i32;
}
"#,
            r#"
mod some_module {
    pub type Result<T> = core::result::Result<T, ()>;
}

use some_module::Result;

fn foo() -> Result<i32> {
    return Ok(42i32);
}
"#,
            WrapperKind::Result.label(),
        );

        check_assist_by_label(
            wrap_return_type,
            r#"
//- minicore: result
mod some_module {
    pub type Result<T> = core::result::Result<T, ()>;
}

use some_module::*;

fn foo() -> i3$02 {
    return 42i32;
}
"#,
            r#"
mod some_module {
    pub type Result<T> = core::result::Result<T, ()>;
}

use some_module::*;

fn foo() -> Result<i32> {
    return Ok(42i32);
}
"#,
            WrapperKind::Result.label(),
        );
    }

    #[test]
    fn wrap_return_type_in_local_result_type_from_function_body() {
        check_assist_by_label(
            wrap_return_type,
            r#"
//- minicore: result
fn foo() -> i3$02 {
    type Result<T> = core::result::Result<T, ()>;
    0
}
"#,
            r#"
fn foo() -> Result<i32, ${0:_}> {
    type Result<T> = core::result::Result<T, ()>;
    Ok(0)
}
"#,
            WrapperKind::Result.label(),
        );
    }

    #[test]
    fn wrap_return_type_in_local_result_type_already_using_alias() {
        check_assist_not_applicable_by_label(
            wrap_return_type,
            r#"
//- minicore: result
pub type Result<T> = core::result::Result<T, ()>;

fn foo() -> Result<i3$02> {
    return Ok(42i32);
}
"#,
            WrapperKind::Result.label(),
        );
    }

    #[test]
    fn wrap_return_type_in_local_result_type_multiple_generics() {
        check_assist_by_label(
            wrap_return_type,
            r#"
//- minicore: result
type Result<T, E> = core::result::Result<T, E>;

fn foo() -> i3$02 {
    0
}
"#,
            r#"
type Result<T, E> = core::result::Result<T, E>;

fn foo() -> Result<i32, ${0:_}> {
    Ok(0)
}
"#,
            WrapperKind::Result.label(),
        );

        check_assist_by_label(
            wrap_return_type,
            r#"
//- minicore: result
type Result<T, E> = core::result::Result<Foo<T, E>, ()>;

fn foo() -> i3$02 {
    0
}
            "#,
            r#"
type Result<T, E> = core::result::Result<Foo<T, E>, ()>;

fn foo() -> Result<i32, ${0:_}> {
    Ok(0)
}
            "#,
            WrapperKind::Result.label(),
        );

        check_assist_by_label(
            wrap_return_type,
            r#"
//- minicore: result
type Result<'a, T, E> = core::result::Result<Foo<T, E>, &'a ()>;

fn foo() -> i3$02 {
    0
}
            "#,
            r#"
type Result<'a, T, E> = core::result::Result<Foo<T, E>, &'a ()>;

fn foo() -> Result<'_, i32, ${0:_}> {
    Ok(0)
}
            "#,
            WrapperKind::Result.label(),
        );

        check_assist_by_label(
            wrap_return_type,
            r#"
//- minicore: result
type Result<T, const N: usize> = core::result::Result<Foo<T>, Bar<N>>;

fn foo() -> i3$02 {
    0
}
            "#,
            r#"
type Result<T, const N: usize> = core::result::Result<Foo<T>, Bar<N>>;

fn foo() -> Result<i32, ${0:_}> {
    Ok(0)
}
            "#,
            WrapperKind::Result.label(),
        );
    }

    #[test]
    fn already_wrapped() {
        check_assist_by_label(
            wrap_return_type,
            r#"
//- minicore: option
fn foo() -> i32$0 {
    if false {
        0
    } else {
        Some(1)
    }
}
            "#,
            r#"
fn foo() -> Option<i32> {
    if false {
        Some(0)
    } else {
        Some(1)
    }
}
            "#,
            WrapperKind::Option.label(),
        );
        check_assist_by_label(
            wrap_return_type,
            r#"
//- minicore: result
fn foo() -> i32$0 {
    if false {
        0
    } else {
        Ok(1)
    }
}
            "#,
            r#"
fn foo() -> Result<i32, ${0:_}> {
    if false {
        Ok(0)
    } else {
        Ok(1)
    }
}
            "#,
            WrapperKind::Result.label(),
        );
    }
}
