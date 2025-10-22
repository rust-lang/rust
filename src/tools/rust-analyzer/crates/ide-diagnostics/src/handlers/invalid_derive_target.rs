use crate::{Diagnostic, DiagnosticCode, DiagnosticsContext};

// Diagnostic: invalid-derive-target
//
// This diagnostic is shown when the derive attribute is used on an item other than a `struct`,
// `enum` or `union`.
pub(crate) fn invalid_derive_target(
    ctx: &DiagnosticsContext<'_>,
    d: &hir::InvalidDeriveTarget,
) -> Diagnostic {
    let display_range = ctx.sema.diagnostics_display_range(d.node);

    Diagnostic::new(
        DiagnosticCode::RustcHardError("E0774"),
        "`derive` may only be applied to `struct`s, `enum`s and `union`s",
        display_range,
    )
    .stable()
}

#[cfg(test)]
mod tests {
    use crate::tests::check_diagnostics;

    #[test]
    fn fails_on_function() {
        check_diagnostics(
            r#"
//- minicore:derive
mod __ {
    #[derive()]
  //^^^^^^^^^^^ error: `derive` may only be applied to `struct`s, `enum`s and `union`s
    fn main() {}
}
            "#,
        );
    }
}
