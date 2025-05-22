use crate::{Diagnostic, DiagnosticCode, DiagnosticsContext};

// Diagnostic: unresolved-ident
//
// This diagnostic is triggered if an expr-position ident is invalid.
pub(crate) fn unresolved_ident(
    ctx: &DiagnosticsContext<'_>,
    d: &hir::UnresolvedIdent,
) -> Diagnostic {
    let mut range =
        ctx.sema.diagnostics_display_range(d.node.map(|(node, _)| node.syntax_node_ptr()));
    if let Some(in_node_range) = d.node.value.1 {
        range.range = in_node_range + range.range.start();
    }
    Diagnostic::new(DiagnosticCode::RustcHardError("E0425"), "no such value in this scope", range)
}

#[cfg(test)]
mod tests {
    use crate::tests::check_diagnostics;

    #[test]
    fn feature() {
        check_diagnostics(
            r#"
//- minicore: fmt
fn main() {
    format_args!("{unresolved}");
                // ^^^^^^^^^^ error: no such value in this scope
}
"#,
        )
    }

    #[test]
    fn missing() {
        check_diagnostics(
            r#"
fn main() {
    let _ = x;
          //^ error: no such value in this scope
}
"#,
        );
    }

    #[test]
    fn present() {
        check_diagnostics(
            r#"
fn main() {
    let x = 5;
    let _ = x;
}
"#,
        );
    }

    #[test]
    fn unresolved_self_val() {
        check_diagnostics(
            r#"
fn main() {
    self.a;
  //^^^^ error: no such value in this scope
    let self:
         self =
            self;
          //^^^^ error: no such value in this scope
}
"#,
        );
    }
}
