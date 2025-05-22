use hir::HirDisplay;

use crate::{Diagnostic, DiagnosticCode, DiagnosticsContext};

// Diagnostic: expected-function
//
// This diagnostic is triggered if a call is made on something that is not callable.
pub(crate) fn expected_function(
    ctx: &DiagnosticsContext<'_>,
    d: &hir::ExpectedFunction,
) -> Diagnostic {
    Diagnostic::new_with_syntax_node_ptr(
        ctx,
        DiagnosticCode::RustcHardError("E0618"),
        format!("expected function, found {}", d.found.display(ctx.sema.db, ctx.display_target)),
        d.call.map(|it| it.into()),
    )
}

#[cfg(test)]
mod tests {
    use crate::tests::check_diagnostics;

    #[test]
    fn smoke_test() {
        check_diagnostics(
            r#"
fn foo() {
    let x = 3;
    x();
 // ^^^ error: expected function, found i32
    ""();
 // ^^^^ error: expected function, found &'static str
    foo();
}
"#,
        );
    }

    #[test]
    fn no_error_for_async_fn_traits() {
        check_diagnostics(
            r#"
//- minicore: async_fn
async fn f(it: impl AsyncFn(u32) -> i32) {
    let fut = it(0);
    let _: i32 = fut.await;
}
async fn g(mut it: impl AsyncFnMut(u32) -> i32) {
    let fut = it(0);
    let _: i32 = fut.await;
}
async fn h(it: impl AsyncFnOnce(u32) -> i32) {
    let fut = it(0);
    let _: i32 = fut.await;
}
        "#,
        );
    }
}
