use hir::{db::DefDatabase, HirDisplay};

use crate::{Diagnostic, DiagnosticsContext};

// Diagnostic: not-implemented
//
// This diagnostic is triggered if a type doesn't implement a necessary trait.
pub(crate) fn not_implemented(ctx: &DiagnosticsContext<'_>, d: &hir::NotImplemented) -> Diagnostic {
    Diagnostic::new(
        "not-implemented",
        format!(
            "the trait `{}` is not implemented for `{}`",
            ctx.sema.db.trait_data(d.trait_).name,
            d.ty.display(ctx.sema.db)
        ),
        ctx.sema.diagnostics_display_range(d.expr.clone().map(|it| it.into())).range,
    )
}

#[cfg(test)]
mod tests {
    use crate::tests::check_diagnostics;

    #[test]
    fn missing_try_impl() {
        check_diagnostics(
            r#"
//- minicore: try
fn main() {
    ()?;
} //^^ error: the trait `Try` is not implemented for `()`
"#,
        )
    }
}
