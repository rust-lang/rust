use crate::{Diagnostic, DiagnosticCode, DiagnosticsContext};

// Diagnostic: return-outside-of-function
//
// This diagnostic triggers if return or become is used outside of a function body.
pub(crate) fn return_outside_function(
    ctx: &DiagnosticsContext<'_, '_>,
    d: &hir::ReturnOutsideFunction,
) -> Diagnostic {
    let construct = match d.kind {
        hir::ReturnKind::ReturnExpr => "return",
        hir::ReturnKind::BecomeExpr => "become",
    };
    Diagnostic::new_with_syntax_node_ptr(
        ctx,
        DiagnosticCode::RustcHardError("E0572"),
        format!("{construct} statement outside of function body"),
        d.expr.map(|it| it.into()),
    )
}

#[cfg(test)]
mod tests {
    use crate::tests::check_diagnostics;

    #[test]
    fn return_in_const() {
        check_diagnostics(
            r#"
const _: () = {
    return;
  //^^^^^^ error: return statement outside of function body
};
"#,
        );
    }

    #[test]
    fn return_in_static() {
        check_diagnostics(
            r#"
static _S: i32 = {
    return 0;
  //^^^^^^^^ error: return statement outside of function body
    0
};
"#,
        );
    }

    #[test]
    fn return_in_function_is_correct() {
        check_diagnostics(
            r#"
fn foo() -> i32 {
    if true { return 42; }
    0
}
"#,
        );
    }

    #[test]
    fn become_in_const() {
        check_diagnostics(
            r#"
const _: () = {
    become 0;
  //^^^^^^^^ error: become statement outside of function body
};
"#,
        );
    }

    #[test]
    fn become_in_static() {
        check_diagnostics(
            r#"
static _S: () = {
    become 0;
  //^^^^^^^^ error: become statement outside of function body
    ()
};
"#,
        );
    }

    #[test]
    fn become_in_function_is_correct() {
        check_diagnostics(
            r#"
fn foo() {
    if true { become (); }
}
"#,
        );
    }
}
