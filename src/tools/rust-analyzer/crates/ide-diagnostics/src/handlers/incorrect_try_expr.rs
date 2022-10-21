use hir::InFile;

use crate::{Diagnostic, DiagnosticsContext};

// Diagnostic: incorrect-try-target
//
// This diagnostic is triggered if a question mark operator was used in a context where it is not applicable.
pub(crate) fn incorrect_try_expr(
    ctx: &DiagnosticsContext<'_>,
    d: &hir::IncorrectTryExpr,
) -> Diagnostic {
    Diagnostic::new(
        "incorrect-try-target",
        format!("the return type of the containing function does not implement `FromResidual`"),
        ctx.sema
            .diagnostics_display_range(InFile::new(d.expr.file_id, d.expr.value.clone().into()))
            .range,
    )
}

#[cfg(test)]
mod tests {
    use crate::tests::check_diagnostics;

    #[test]
    fn try_ops_diag() {
        check_diagnostics(
            r#"
//- minicore: try
fn test() {
    core::ops::ControlFlow::<u32, f32>::Continue(1.0)?;
 // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ error: the return type of the containing function does not implement `FromResidual`
}
"#,
        );
    }
}
