use hir::HirDisplay;

use crate::{Diagnostic, DiagnosticCode, DiagnosticsContext};

// Diagnostic: cannot-implicitly-deref-trait-object
//
// This diagnostic is triggered when a pointer to a trait object is implicitly
// dereferenced by a pattern.
pub(crate) fn cannot_implicitly_deref_trait_object(
    ctx: &DiagnosticsContext<'_, '_>,
    d: &hir::CannotImplicitlyDerefTraitObject<'_>,
) -> Diagnostic {
    Diagnostic::new_with_syntax_node_ptr(
        ctx,
        DiagnosticCode::RustcHardError("E0033"),
        format!(
            "type `{}` cannot be dereferenced",
            d.found.display(ctx.sema.db, ctx.display_target)
        ),
        d.pat.map(Into::into),
    )
    .stable()
}

#[cfg(test)]
mod tests {
    use crate::tests::check_diagnostics;

    #[test]
    fn trait_object_pattern_deref() {
        check_diagnostics(
            r#"
trait Trait {}

fn f(x: &dyn Trait) {
    let &ref _y = x;
      //^^^^^^^ error: type `&(dyn Trait + 'static)` cannot be dereferenced
}
"#,
        );
    }

    #[test]
    fn allows_sized_ref_pattern_deref() {
        check_diagnostics(
            r#"
fn f(x: &i32) {
    let &ref _y = x;
}
"#,
        );
    }
}
