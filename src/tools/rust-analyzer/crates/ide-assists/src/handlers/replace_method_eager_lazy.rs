use hir::Semantics;
use ide_db::{RootDatabase, assists::AssistId, defs::Definition};
use syntax::{
    AstNode,
    ast::{self, Expr, HasArgList, make},
};

use crate::{AssistContext, Assists};

// Assist: replace_with_lazy_method
//
// Replace `unwrap_or` with `unwrap_or_else` and `ok_or` with `ok_or_else`.
//
// ```
// # //- minicore:option, fn
// fn foo() {
//     let a = Some(1);
//     a.unwra$0p_or(2);
// }
// ```
// ->
// ```
// fn foo() {
//     let a = Some(1);
//     a.unwrap_or_else(|| 2);
// }
// ```
pub(crate) fn replace_with_lazy_method(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    let call: ast::MethodCallExpr = ctx.find_node_at_offset()?;
    let scope = ctx.sema.scope(call.syntax())?;

    let last_arg = call.arg_list()?.args().next()?;
    let method_name = call.name_ref()?;

    let callable = ctx.sema.resolve_method_call_as_callable(&call)?;
    let (_, receiver_ty) = callable.receiver_param(ctx.sema.db)?;
    let n_params = callable.n_params() + 1;

    let method_name_lazy = lazy_method_name(&method_name.text());

    receiver_ty.iterate_method_candidates_with_traits(
        ctx.sema.db,
        &scope,
        &scope.visible_traits().0,
        None,
        |func| {
            let valid = func.name(ctx.sema.db).as_str() == &*method_name_lazy
                && func.num_params(ctx.sema.db) == n_params
                && {
                    let params = func.params_without_self(ctx.sema.db);
                    let last_p = params.first()?;
                    // FIXME: Check that this has the form of `() -> T` where T is the current type of the argument
                    last_p.ty().impls_fnonce(ctx.sema.db)
                };
            valid.then_some(func)
        },
    )?;

    acc.add(
        AssistId::refactor_rewrite("replace_with_lazy_method"),
        format!("Replace {method_name} with {method_name_lazy}"),
        call.syntax().text_range(),
        |builder| {
            let closured = into_closure(&last_arg, &method_name_lazy);
            builder.replace(method_name.syntax().text_range(), method_name_lazy);
            builder.replace_ast(last_arg, closured);
        },
    )
}

fn lazy_method_name(name: &str) -> String {
    if ends_is(name, "or") {
        format!("{name}_else")
    } else if ends_is(name, "and") {
        format!("{name}_then")
    } else if ends_is(name, "then_some") {
        name.strip_suffix("_some").unwrap().to_owned()
    } else {
        format!("{name}_with")
    }
}

fn into_closure(param: &Expr, name_lazy: &str) -> Expr {
    (|| {
        if let ast::Expr::CallExpr(call) = param {
            if call.arg_list()?.args().count() == 0 { Some(call.expr()?) } else { None }
        } else {
            None
        }
    })()
    .unwrap_or_else(|| {
        let pats = (name_lazy == "and_then")
            .then(|| make::untyped_param(make::ext::simple_ident_pat(make::name("it")).into()));
        make::expr_closure(pats, param.clone()).into()
    })
}

// Assist: replace_with_eager_method
//
// Replace `unwrap_or_else` with `unwrap_or` and `ok_or_else` with `ok_or`.
//
// ```
// # //- minicore:option, fn
// fn foo() {
//     let a = Some(1);
//     a.unwra$0p_or_else(|| 2);
// }
// ```
// ->
// ```
// fn foo() {
//     let a = Some(1);
//     a.unwrap_or(2);
// }
// ```
pub(crate) fn replace_with_eager_method(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    let call: ast::MethodCallExpr = ctx.find_node_at_offset()?;
    let scope = ctx.sema.scope(call.syntax())?;

    let last_arg = call.arg_list()?.args().next()?;
    let method_name = call.name_ref()?;

    let callable = ctx.sema.resolve_method_call_as_callable(&call)?;
    let (_, receiver_ty) = callable.receiver_param(ctx.sema.db)?;
    let n_params = callable.n_params() + 1;
    let params = callable.params();

    // FIXME: Check that the arg is of the form `() -> T`
    if !params.first()?.ty().impls_fnonce(ctx.sema.db) {
        return None;
    }

    let method_name_text = method_name.text();
    let method_name_eager = eager_method_name(&method_name_text)?;

    receiver_ty.iterate_method_candidates_with_traits(
        ctx.sema.db,
        &scope,
        &scope.visible_traits().0,
        None,
        |func| {
            let valid = func.name(ctx.sema.db).as_str() == method_name_eager
                && func.num_params(ctx.sema.db) == n_params;
            valid.then_some(func)
        },
    )?;

    acc.add(
        AssistId::refactor_rewrite("replace_with_eager_method"),
        format!("Replace {method_name} with {method_name_eager}"),
        call.syntax().text_range(),
        |builder| {
            builder.replace(method_name.syntax().text_range(), method_name_eager);
            let called = into_call(&last_arg, &ctx.sema);
            builder.replace_ast(last_arg, called);
        },
    )
}

fn into_call(param: &Expr, sema: &Semantics<'_, RootDatabase>) -> Expr {
    (|| {
        if let ast::Expr::ClosureExpr(closure) = param {
            let mut params = closure.param_list()?.params();
            match params.next() {
                Some(_) if params.next().is_none() => {
                    let params = sema.resolve_expr_as_callable(param)?.params();
                    let used_param = Definition::Local(params.first()?.as_local(sema.db)?)
                        .usages(sema)
                        .at_least_one();
                    if used_param { None } else { Some(closure.body()?) }
                }
                None => Some(closure.body()?),
                Some(_) => None,
            }
        } else {
            None
        }
    })()
    .unwrap_or_else(|| {
        let callable = if needs_parens_in_call(param) {
            make::expr_paren(param.clone()).into()
        } else {
            param.clone()
        };
        make::expr_call(callable, make::arg_list(Vec::new())).into()
    })
}

fn eager_method_name(name: &str) -> Option<&str> {
    if name == "then" {
        return Some("then_some");
    }

    name.strip_suffix("_else")
        .or_else(|| name.strip_suffix("_then"))
        .or_else(|| name.strip_suffix("_with"))
}

fn ends_is(name: &str, end: &str) -> bool {
    name.strip_suffix(end).is_some_and(|s| s.is_empty() || s.ends_with('_'))
}

fn needs_parens_in_call(param: &Expr) -> bool {
    let call = make::expr_call(make::ext::expr_unit(), make::arg_list(Vec::new()));
    let callable = call.expr().expect("invalid make call");
    param.needs_parens_in_place_of(call.syntax(), callable.syntax())
}

#[cfg(test)]
mod tests {
    use crate::tests::check_assist;

    use super::*;

    #[test]
    fn replace_or_with_or_else_simple() {
        check_assist(
            replace_with_lazy_method,
            r#"
//- minicore: option, fn
fn foo() {
    let foo = Some(1);
    return foo.unwrap_$0or(2);
}
"#,
            r#"
fn foo() {
    let foo = Some(1);
    return foo.unwrap_or_else(|| 2);
}
"#,
        )
    }

    #[test]
    fn replace_or_with_or_else_call() {
        check_assist(
            replace_with_lazy_method,
            r#"
//- minicore: option, fn
fn foo() {
    let foo = Some(1);
    return foo.unwrap_$0or(x());
}
"#,
            r#"
fn foo() {
    let foo = Some(1);
    return foo.unwrap_or_else(x);
}
"#,
        )
    }

    #[test]
    fn replace_or_with_or_else_block() {
        check_assist(
            replace_with_lazy_method,
            r#"
//- minicore: option, fn
fn foo() {
    let foo = Some(1);
    return foo.unwrap_$0or({
        let mut x = bar();
        for i in 0..10 {
            x += i;
        }
        x
    });
}
"#,
            r#"
fn foo() {
    let foo = Some(1);
    return foo.unwrap_or_else(|| {
        let mut x = bar();
        for i in 0..10 {
            x += i;
        }
        x
    });
}
"#,
        )
    }

    #[test]
    fn replace_or_else_with_or_simple() {
        check_assist(
            replace_with_eager_method,
            r#"
//- minicore: option, fn
fn foo() {
    let foo = Some(1);
    return foo.unwrap_$0or_else(|| 2);
}
"#,
            r#"
fn foo() {
    let foo = Some(1);
    return foo.unwrap_or(2);
}
"#,
        )
    }

    #[test]
    fn replace_or_else_with_or_call() {
        check_assist(
            replace_with_eager_method,
            r#"
//- minicore: option, fn
fn foo() {
    let foo = Some(1);
    return foo.unwrap_$0or_else(x);
}

fn x() -> i32 { 0 }
"#,
            r#"
fn foo() {
    let foo = Some(1);
    return foo.unwrap_or(x());
}

fn x() -> i32 { 0 }
"#,
        )
    }

    #[test]
    fn replace_or_else_with_or_map() {
        check_assist(
            replace_with_eager_method,
            r#"
//- minicore: option, fn
fn foo() {
    let foo = Some("foo");
    return foo.map$0_or_else(|| 42, |v| v.len());
}
"#,
            r#"
fn foo() {
    let foo = Some("foo");
    return foo.map_or(42, |v| v.len());
}
"#,
        )
    }

    #[test]
    fn replace_and_with_and_then() {
        check_assist(
            replace_with_lazy_method,
            r#"
//- minicore: option, fn
fn foo() {
    let foo = Some("foo");
    return foo.and$0(Some("bar"));
}
"#,
            r#"
fn foo() {
    let foo = Some("foo");
    return foo.and_then(|it| Some("bar"));
}
"#,
        )
    }

    #[test]
    fn replace_and_then_with_and() {
        check_assist(
            replace_with_eager_method,
            r#"
//- minicore: option, fn
fn foo() {
    let foo = Some("foo");
    return foo.and_then$0(|it| Some("bar"));
}
"#,
            r#"
fn foo() {
    let foo = Some("foo");
    return foo.and(Some("bar"));
}
"#,
        )
    }

    #[test]
    fn replace_and_then_with_and_used_param() {
        check_assist(
            replace_with_eager_method,
            r#"
//- minicore: option, fn
fn foo() {
    let foo = Some("foo");
    return foo.and_then$0(|it| Some(it.strip_suffix("bar")));
}
"#,
            r#"
fn foo() {
    let foo = Some("foo");
    return foo.and((|it| Some(it.strip_suffix("bar")))());
}
"#,
        )
    }

    #[test]
    fn replace_then_some_with_then() {
        check_assist(
            replace_with_lazy_method,
            r#"
//- minicore: option, fn, bool_impl
fn foo() {
    let foo = true;
    let x = foo.then_some$0(2);
}
"#,
            r#"
fn foo() {
    let foo = true;
    let x = foo.then(|| 2);
}
"#,
        )
    }

    #[test]
    fn replace_then_with_then_some() {
        check_assist(
            replace_with_eager_method,
            r#"
//- minicore: option, fn, bool_impl
fn foo() {
    let foo = true;
    let x = foo.then$0(|| 2);
}
"#,
            r#"
fn foo() {
    let foo = true;
    let x = foo.then_some(2);
}
"#,
        )
    }

    #[test]
    fn replace_then_with_then_some_needs_parens() {
        check_assist(
            replace_with_eager_method,
            r#"
//- minicore: option, fn, bool_impl
struct Func { f: fn() -> i32 }
fn foo() {
    let foo = true;
    let func = Func { f: || 2 };
    let x = foo.then$0(func.f);
}
"#,
            r#"
struct Func { f: fn() -> i32 }
fn foo() {
    let foo = true;
    let func = Func { f: || 2 };
    let x = foo.then_some((func.f)());
}
"#,
        )
    }
}
