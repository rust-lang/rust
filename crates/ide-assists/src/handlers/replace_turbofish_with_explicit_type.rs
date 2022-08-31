use hir::HirDisplay;
use syntax::{
    ast::{Expr, GenericArg, GenericArgList},
    ast::{LetStmt, Type::InferType},
    AstNode, TextRange,
};

use crate::{
    assist_context::{AssistContext, Assists},
    AssistId, AssistKind,
};

// Assist: replace_turbofish_with_explicit_type
//
// Converts `::<_>` to an explicit type assignment.
//
// ```
// fn make<T>() -> T { ) }
// fn main() {
//     let a = make$0::<i32>();
// }
// ```
// ->
// ```
// fn make<T>() -> T { ) }
// fn main() {
//     let a: i32 = make();
// }
// ```
pub(crate) fn replace_turbofish_with_explicit_type(
    acc: &mut Assists,
    ctx: &AssistContext<'_>,
) -> Option<()> {
    let let_stmt = ctx.find_node_at_offset::<LetStmt>()?;

    let initializer = let_stmt.initializer()?;

    let generic_args = generic_arg_list(&initializer)?;

    // Find range of ::<_>
    let colon2 = generic_args.coloncolon_token()?;
    let r_angle = generic_args.r_angle_token()?;
    let turbofish_range = TextRange::new(colon2.text_range().start(), r_angle.text_range().end());

    let turbofish_args: Vec<GenericArg> = generic_args.generic_args().into_iter().collect();

    // Find type of ::<_>
    if turbofish_args.len() != 1 {
        cov_mark::hit!(not_applicable_if_not_single_arg);
        return None;
    }

    // An improvement would be to check that this is correctly part of the return value of the
    // function call, or sub in the actual return type.
    let returned_type = match ctx.sema.type_of_expr(&initializer) {
        Some(returned_type) if !returned_type.original.contains_unknown() => {
            let module = ctx.sema.scope(let_stmt.syntax())?.module();
            returned_type.original.display_source_code(ctx.db(), module.into()).ok()?
        }
        _ => {
            cov_mark::hit!(fallback_to_turbofish_type_if_type_info_not_available);
            turbofish_args[0].to_string()
        }
    };

    let initializer_start = initializer.syntax().text_range().start();
    if ctx.offset() > turbofish_range.end() || ctx.offset() < initializer_start {
        cov_mark::hit!(not_applicable_outside_turbofish);
        return None;
    }

    if let None = let_stmt.colon_token() {
        // If there's no colon in a let statement, then there is no explicit type.
        // let x = fn::<...>();
        let ident_range = let_stmt.pat()?.syntax().text_range();

        return acc.add(
            AssistId("replace_turbofish_with_explicit_type", AssistKind::RefactorRewrite),
            "Replace turbofish with explicit type",
            TextRange::new(initializer_start, turbofish_range.end()),
            |builder| {
                builder.insert(ident_range.end(), format!(": {}", returned_type));
                builder.delete(turbofish_range);
            },
        );
    } else if let Some(InferType(t)) = let_stmt.ty() {
        // If there's a type inference underscore, we can offer to replace it with the type in
        // the turbofish.
        // let x: _ = fn::<...>();
        let underscore_range = t.syntax().text_range();

        return acc.add(
            AssistId("replace_turbofish_with_explicit_type", AssistKind::RefactorRewrite),
            "Replace `_` with turbofish type",
            turbofish_range,
            |builder| {
                builder.replace(underscore_range, returned_type);
                builder.delete(turbofish_range);
            },
        );
    }

    None
}

fn generic_arg_list(expr: &Expr) -> Option<GenericArgList> {
    match expr {
        Expr::MethodCallExpr(expr) => expr.generic_arg_list(),
        Expr::CallExpr(expr) => {
            if let Expr::PathExpr(pe) = expr.expr()? {
                pe.path()?.segment()?.generic_arg_list()
            } else {
                cov_mark::hit!(not_applicable_if_non_path_function_call);
                return None;
            }
        }
        Expr::AwaitExpr(expr) => generic_arg_list(&expr.expr()?),
        Expr::TryExpr(expr) => generic_arg_list(&expr.expr()?),
        _ => {
            cov_mark::hit!(not_applicable_if_non_function_call_initializer);
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::tests::{check_assist, check_assist_not_applicable, check_assist_target};

    #[test]
    fn replaces_turbofish_for_vec_string() {
        cov_mark::check!(fallback_to_turbofish_type_if_type_info_not_available);
        check_assist(
            replace_turbofish_with_explicit_type,
            r#"
fn make<T>() -> T {}
fn main() {
    let a = make$0::<Vec<String>>();
}
"#,
            r#"
fn make<T>() -> T {}
fn main() {
    let a: Vec<String> = make();
}
"#,
        );
    }

    #[test]
    fn replaces_method_calls() {
        // foo.make() is a method call which uses a different expr in the let initializer
        cov_mark::check!(fallback_to_turbofish_type_if_type_info_not_available);
        check_assist(
            replace_turbofish_with_explicit_type,
            r#"
fn make<T>() -> T {}
fn main() {
    let a = foo.make$0::<Vec<String>>();
}
"#,
            r#"
fn make<T>() -> T {}
fn main() {
    let a: Vec<String> = foo.make();
}
"#,
        );
    }

    #[test]
    fn replace_turbofish_target() {
        check_assist_target(
            replace_turbofish_with_explicit_type,
            r#"
fn make<T>() -> T {}
fn main() {
    let a = $0make::<Vec<String>>();
}
"#,
            r#"make::<Vec<String>>"#,
        );
    }

    #[test]
    fn not_applicable_outside_turbofish() {
        cov_mark::check!(not_applicable_outside_turbofish);
        check_assist_not_applicable(
            replace_turbofish_with_explicit_type,
            r#"
fn make<T>() -> T {}
fn main() {
    let $0a = make::<Vec<String>>();
}
"#,
        );
    }

    #[test]
    fn replace_inferred_type_placeholder() {
        check_assist(
            replace_turbofish_with_explicit_type,
            r#"
fn make<T>() -> T {}
fn main() {
    let a: _ = make$0::<Vec<String>>();
}
"#,
            r#"
fn make<T>() -> T {}
fn main() {
    let a: Vec<String> = make();
}
"#,
        );
    }

    #[test]
    fn not_applicable_constant_initializer() {
        cov_mark::check!(not_applicable_if_non_function_call_initializer);
        check_assist_not_applicable(
            replace_turbofish_with_explicit_type,
            r#"
fn make<T>() -> T {}
fn main() {
    let a = "foo"$0;
}
"#,
        );
    }

    #[test]
    fn not_applicable_non_path_function_call() {
        cov_mark::check!(not_applicable_if_non_path_function_call);
        check_assist_not_applicable(
            replace_turbofish_with_explicit_type,
            r#"
fn make<T>() -> T {}
fn main() {
    $0let a = (|| {})();
}
"#,
        );
    }

    #[test]
    fn non_applicable_multiple_generic_args() {
        cov_mark::check!(not_applicable_if_not_single_arg);
        check_assist_not_applicable(
            replace_turbofish_with_explicit_type,
            r#"
fn make<T>() -> T {}
fn main() {
    let a = make$0::<Vec<String>, i32>();
}
"#,
        );
    }

    #[test]
    fn replaces_turbofish_for_known_type() {
        check_assist(
            replace_turbofish_with_explicit_type,
            r#"
fn make<T>() -> T {}
fn main() {
    let a = make$0::<i32>();
}
"#,
            r#"
fn make<T>() -> T {}
fn main() {
    let a: i32 = make();
}
"#,
        );
        check_assist(
            replace_turbofish_with_explicit_type,
            r#"
//- minicore: option
fn make<T>() -> T {}
fn main() {
    let a = make$0::<Option<bool>>();
}
"#,
            r#"
fn make<T>() -> T {}
fn main() {
    let a: Option<bool> = make();
}
"#,
        );
    }

    #[test]
    fn replaces_turbofish_not_same_type() {
        check_assist(
            replace_turbofish_with_explicit_type,
            r#"
//- minicore: option
fn make<T>() -> Option<T> {}
fn main() {
    let a = make$0::<u128>();
}
"#,
            r#"
fn make<T>() -> Option<T> {}
fn main() {
    let a: Option<u128> = make();
}
"#,
        );
    }

    #[test]
    fn replaces_turbofish_for_type_with_defaulted_generic_param() {
        check_assist(
            replace_turbofish_with_explicit_type,
            r#"
struct HasDefault<T, U = i32>(T, U);
fn make<T>() -> HasDefault<T> {}
fn main() {
    let a = make$0::<bool>();
}
"#,
            r#"
struct HasDefault<T, U = i32>(T, U);
fn make<T>() -> HasDefault<T> {}
fn main() {
    let a: HasDefault<bool> = make();
}
"#,
        );
    }

    #[test]
    fn replaces_turbofish_try_await() {
        check_assist(
            replace_turbofish_with_explicit_type,
            r#"
//- minicore: option, future
struct Fut<T>(T);
impl<T> core::future::Future for Fut<T> {
    type Output = Option<T>;
}
fn make<T>() -> Fut<T> {}
fn main() {
    let a = make$0::<bool>().await?;
}
"#,
            r#"
struct Fut<T>(T);
impl<T> core::future::Future for Fut<T> {
    type Output = Option<T>;
}
fn make<T>() -> Fut<T> {}
fn main() {
    let a: bool = make().await?;
}
"#,
        );
    }
}
