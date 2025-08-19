use crate::{Diagnostic, DiagnosticsContext, adjusted_display_range};

// Diagnostic: await-outside-of-async
//
// This diagnostic is triggered if the `await` keyword is used outside of an async function or block
pub(crate) fn await_outside_of_async(
    ctx: &DiagnosticsContext<'_>,
    d: &hir::AwaitOutsideOfAsync,
) -> Diagnostic {
    let display_range =
        adjusted_display_range(ctx, d.node, &|node| Some(node.await_token()?.text_range()));
    Diagnostic::new(
        crate::DiagnosticCode::RustcHardError("E0728"),
        format!("`await` is used inside {}, which is not an `async` context", d.location),
        display_range,
    )
    .stable()
}

#[cfg(test)]
mod tests {
    use crate::tests::check_diagnostics;

    #[test]
    fn await_inside_non_async_fn() {
        check_diagnostics(
            r#"
async fn foo() {}

fn bar() {
    foo().await;
        //^^^^^ error: `await` is used inside non-async function, which is not an `async` context
}
"#,
        );
    }

    #[test]
    fn await_inside_async_fn() {
        check_diagnostics(
            r#"
async fn foo() {}

async fn bar() {
    foo().await;
}
"#,
        );
    }

    #[test]
    fn await_inside_closure() {
        check_diagnostics(
            r#"
async fn foo() {}

async fn bar() {
    let _a = || { foo().await };
                      //^^^^^ error: `await` is used inside non-async closure, which is not an `async` context
}
"#,
        );
    }

    #[test]
    fn await_inside_async_block() {
        check_diagnostics(
            r#"
async fn foo() {}

fn bar() {
    let _a = async { foo().await };
}
"#,
        );
    }

    #[test]
    fn await_in_complex_context() {
        check_diagnostics(
            r#"
async fn foo() {}

fn bar() {
    async fn baz() {
        let a = foo().await;
    }

    let x = || {
        let y = async {
            baz().await;
            let z = || {
                baz().await;
                    //^^^^^ error: `await` is used inside non-async closure, which is not an `async` context
            };
        };
    };
}
"#,
        );
    }
}
