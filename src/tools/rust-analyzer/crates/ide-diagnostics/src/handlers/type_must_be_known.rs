use either::Either;
use hir::{HirDisplay, SpanAst};
use stdx::format_to;
use syntax::{AstNode, SyntaxNodePtr, ast};

use crate::{Diagnostic, DiagnosticCode, DiagnosticsContext};

// Diagnostic: type-must-be-known
//
// This diagnostic is triggered when rust-analyzer cannot infer some type.
pub(crate) fn type_must_be_known<'db>(
    ctx: &DiagnosticsContext<'db, '_>,
    d: &hir::TypeMustBeKnown<'db>,
) -> Diagnostic {
    let mut at_point = d.at_point.map(|it| it.syntax_node_ptr());
    let mut top_term = d.top_term.clone();

    // Do some adjustments to the node: FIXME: We should probably do that at the emitting site.
    let node = ctx.sema.to_node(d.at_point);
    if let SpanAst::Expr(expr) = &node
        && let Some(Either::Left(top_ty)) = &d.top_term
        && let Some(expr_ty) = ctx.sema.type_of_expr(expr)
        && expr_ty.original == *top_ty
        && !top_ty.is_unknown()
        && let Some(parent) = expr.syntax().parent().and_then(ast::CallExpr::cast)
        && let Some(callable) = top_ty.as_callable(ctx.db())
        && let ret_ty = callable.return_type()
        && ret_ty.contains_unknown()
    {
        top_term = Some(Either::Left(ret_ty));
        at_point.value = SyntaxNodePtr::new(parent.syntax());
    }

    let message = match &top_term {
        Some(top_term) if !matches!(top_term, Either::Left(ty) if ty.is_unknown()) => {
            let mut message = "type annotations needed\nfull type: `".to_owned();
            match top_term {
                Either::Left(ty) => {
                    format_to!(message, "{}", ty.display(ctx.db(), ctx.display_target))
                }
                Either::Right(konst) => message.push_str(konst),
            }
            message.push_str("`\n");
            message
        }
        Some(_) => "type annotations needed".to_owned(),
        None => "type annotations needed; type must be known at this point".to_owned(),
    };
    Diagnostic::new_with_syntax_node_ptr(
        ctx,
        DiagnosticCode::RustcHardError("E0282"),
        message,
        at_point,
    )
}

#[cfg(test)]
mod tests {
    use crate::tests::check_diagnostics;

    #[test]
    fn some_expressions_require_knowing_type() {
        check_diagnostics(
            r#"
fn foo() {
    let var = loop {};
     // ^^^ 💡 warn: unused variable
    var();
 // ^^^ error: type annotations needed; type must be known at this point
    let var = loop {};
     // ^^^ 💡 warn: unused variable
    var[0];
 // ^^^ error: type annotations needed; type must be known at this point
}
        "#,
        );
    }

    #[test]
    fn binding_without_type() {
        check_diagnostics(
            r#"
fn any<T>() -> T { loop {} }
fn foo() {
    let _x = any();
          // ^^^^^ error: type annotations needed
}
        "#,
        );
    }

    #[test]
    fn struct_with_generic() {
        check_diagnostics(
            r#"
struct X<T>(T);
fn any<T>() -> X<T> { loop {} }
fn foo() {
    let _x = any();
          // ^^^^^ error: type annotations needed
              // | full type: `X<{unknown}>`
}
        "#,
        );
    }

    #[test]
    fn const_block_does_not_cause_error() {
        check_diagnostics(
            r#"
fn bar<T>(_inner: fn() -> *const T) {}

fn foo() {
    bar(const { || 0 as *const i32 })
}
        "#,
        );
    }

    #[test]
    fn async_closure_does_not_trigger() {
        check_diagnostics(
            r#"
//- minicore: async_fn
struct Task<R>(R);
fn spawn_in<AsyncFn, R>(_f: AsyncFn) -> Task<R>
where
    R: 'static,
    AsyncFn: AsyncFnOnce(&()) -> R + 'static,
{
    loop {}
}

fn foo() {
    spawn_in(async move |cx| {});
}
        "#,
        );
    }

    #[test]
    fn regression_22263() {
        check_diagnostics(
            r#"
trait From<T> {}
impl<T> From<T> for T {}
#[rustc_reservation_impl = "blah blah"]
impl<T> From<!> for T {}

fn any<T>() -> T {
    loop {}
}
fn foo<T, U: From<T>>(_: T) -> U {
    loop {}
}
fn bar() {
    let _: () = foo(any());
}
        "#,
        );
    }
}
