use crate::{Diagnostic, DiagnosticCode, DiagnosticsContext};

// Diagnostic: mutable-ref
//
// This diagnostic is triggered when binding is taken that is both mutable and by-reference.
pub(crate) fn mutable_ref_binding(
    ctx: &DiagnosticsContext<'_, '_>,
    d: &hir::MutableRefBinding,
) -> Diagnostic {
    Diagnostic::new_with_syntax_node_ptr(
        ctx,
        DiagnosticCode::RustcHardError("E0658"),
        "`mut` bindings cannot also be `ref` by default in 2024 edition",
        d.pat.map(Into::into),
    )
    .stable()
}

#[cfg(test)]
mod tests {
    use crate::tests::check_diagnostics;

    #[test]
    fn mutable_ref_binding_missing_feature() {
        check_diagnostics(
            r#"
//- minicore: option
//- /main.rs 
#![feature(ref_pat_eat_one_layer_2024)]
fn main() {
    let opt_ref = &Some(42);

    if let Some(mut x) = opt_ref {
              //^^^^^ error: `mut` bindings cannot also be `ref` by default in 2024 edition
        x = &5;
    }
}
"#,
        );
    }

    #[test]
    fn mutable_ref_binding_with_feature() {
        check_diagnostics(
            r#"
//- minicore: option
//- /main.rs 
#![feature(ref_pat_eat_one_layer_2024)]
#![feature(mut_ref)]
fn main() {
    let opt_ref = &Some(42);

    if let Some(mut x) = opt_ref {
        x = &5;
    }
}
"#,
        );
    }
}
