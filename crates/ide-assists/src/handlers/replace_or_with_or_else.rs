use ide_db::{
    assists::{AssistId, AssistKind},
    famous_defs::FamousDefs,
};
use syntax::{
    ast::{self, make, Expr, HasArgList},
    AstNode,
};

use crate::{AssistContext, Assists};

// Assist: replace_or_with_or_else
//
// Replace `unwrap_or` with `unwrap_or_else` and `ok_or` with `ok_or_else`.
//
// ```
// let a = Some(1);
// a.unwra$0p_or(2);
// ```
// ->
// ```
// let a = Some(1);
// a.unwrap_or_else(|| 2);
// ```
pub(crate) fn replace_or_with_or_else(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    let call: ast::MethodCallExpr = ctx.find_node_at_offset()?;

    is_option_or_result(call.receiver()?, ctx)?;

    let (name, arg_list) = (call.name_ref()?, call.arg_list()?);

    let replace = match &*name.text() {
        "unwrap_or" => "unwrap_or_else".to_string(),
        "ok_or" => "ok_or_else".to_string(),
        _ => return None,
    };

    let arg = match arg_list.args().collect::<Vec<_>>().as_slice() {
        [] => make::arg_list(Vec::new()),
        [first] => {
            let param = (|| {
                if let ast::Expr::CallExpr(call) = first {
                    if call.arg_list()?.args().count() == 0 {
                        Some(call.expr()?.clone())
                    } else {
                        None
                    }
                } else {
                    None
                }
            })()
            .unwrap_or_else(|| make::expr_closure(None, first.clone()));
            make::arg_list(vec![param])
        }
        _ => return None,
    };

    acc.add(
        AssistId("replace_or_with_or_else", AssistKind::RefactorRewrite),
        "Replace unwrap_or or ok_or with lazy version",
        call.syntax().text_range(),
        |builder| {
            builder.replace(name.syntax().text_range(), replace);
            builder.replace_ast(arg_list, arg)
        },
    )
}

// Assist: replace_or_else_with_or
//
// Replace `unwrap_or_else` with `unwrap_or` and `ok_or_else` with `ok_or`.
//
// ```
// let a = Some(1);
// a.unwra$0p_or_else(|| 2);
// ```
// ->
// ```
// let a = Some(1);
// a.unwrap_or(2);
// ```
pub(crate) fn replace_or_else_with_or(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    let call: ast::MethodCallExpr = ctx.find_node_at_offset()?;

    is_option_or_result(call.receiver()?, ctx)?;

    let (name, arg_list) = (call.name_ref()?, call.arg_list()?);

    let replace = match &*name.text() {
        "unwrap_or_else" => "unwrap_or".to_string(),
        "ok_or_else" => "ok_or".to_string(),
        _ => return None,
    };

    let arg = match arg_list.args().collect::<Vec<_>>().as_slice() {
        [] => make::arg_list(Vec::new()),
        [first] => {
            let param = (|| {
                if let ast::Expr::ClosureExpr(closure) = first {
                    if closure.param_list()?.params().count() == 0 {
                        Some(closure.body()?.clone())
                    } else {
                        None
                    }
                } else {
                    None
                }
            })()
            .unwrap_or_else(|| make::expr_call(first.clone(), make::arg_list(Vec::new())));
            make::arg_list(vec![param])
        }
        _ => return None,
    };

    acc.add(
        AssistId("replace_or_else_with_or", AssistKind::RefactorRewrite),
        "Replace unwrap_or_else or ok_or_else with eager version",
        call.syntax().text_range(),
        |builder| {
            builder.replace(name.syntax().text_range(), replace);
            builder.replace_ast(arg_list, arg)
        },
    )
}

fn is_option_or_result(receiver: Expr, ctx: &AssistContext<'_>) -> Option<()> {
    let ty = ctx.sema.type_of_expr(&receiver)?.adjusted().as_adt()?.as_enum()?;
    let option_enum =
        FamousDefs(&ctx.sema, ctx.sema.scope(receiver.syntax())?.krate()).core_option_Option();

    if let Some(option_enum) = option_enum {
        if ty == option_enum {
            return Some(());
        }
    }

    let result_enum =
        FamousDefs(&ctx.sema, ctx.sema.scope(receiver.syntax())?.krate()).core_result_Result();

    if let Some(result_enum) = result_enum {
        if ty == result_enum {
            return Some(());
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_assist, check_assist_not_applicable};

    use super::*;

    #[test]
    fn replace_or_with_or_else_simple() {
        check_assist(
            replace_or_with_or_else,
            r#"
//- minicore: option
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
            replace_or_with_or_else,
            r#"
//- minicore: option
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
            replace_or_with_or_else,
            r#"
//- minicore: option
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
            replace_or_else_with_or,
            r#"
//- minicore: option
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
            replace_or_else_with_or,
            r#"
//- minicore: option
fn foo() {
    let foo = Some(1);
    return foo.unwrap_$0or_else(x);
}
"#,
            r#"
fn foo() {
    let foo = Some(1);
    return foo.unwrap_or(x());
}
"#,
        )
    }

    #[test]
    fn replace_or_else_with_or_result() {
        check_assist(
            replace_or_else_with_or,
            r#"
//- minicore: result
fn foo() {
    let foo = Ok(1);
    return foo.unwrap_$0or_else(x);
}
"#,
            r#"
fn foo() {
    let foo = Ok(1);
    return foo.unwrap_or(x());
}
"#,
        )
    }

    #[test]
    fn replace_or_else_with_or_not_applicable() {
        check_assist_not_applicable(
            replace_or_else_with_or,
            r#"
fn foo() {
    let foo = Ok(1);
    return foo.unwrap_$0or_else(x);
}
"#,
        )
    }
}
