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
        "bindings cannot be both mutable and by-reference by default in 2024 edition. add experimental #![feature(mut_ref)] for this functionality",
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
#![feature(ref_pat_eat_one_layer_2024)]
struct TestStruct {
   val: i32
}
fn main() {
    let opt_ref = &Some(TestStruct {val: 1});

    if let Some(mut x) = opt_ref {
              //^^^^^ error: bindings cannot be both mutable and by-reference by default in 2024 edition. add experimental #![feature(mut_ref)] for this functionality
        x = &TestStruct{val: 5};
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
#![feature(ref_pat_eat_one_layer_2024)]
#![feature(mut_ref)]
struct TestStruct {
    val: i32
}
fn main() {
    let opt_ref = &Some(TestStruct{val: 1});

    if let Some(mut x) = opt_ref {
        x = &TestStruct{val: 5};
    }
}
"#,
        );
    }
}
