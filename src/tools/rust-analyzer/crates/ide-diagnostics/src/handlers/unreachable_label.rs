use crate::{Diagnostic, DiagnosticCode, DiagnosticsContext};

// Diagnostic: unreachable-label
pub(crate) fn unreachable_label(
    ctx: &DiagnosticsContext<'_>,
    d: &hir::UnreachableLabel,
) -> Diagnostic {
    let name = &d.name;
    Diagnostic::new_with_syntax_node_ptr(
        ctx,
        DiagnosticCode::RustcHardError("E0767"),
        format!("use of unreachable label `{}`", name.display(ctx.sema.db, ctx.edition)),
        d.node.map(|it| it.into()),
    )
    .stable()
}

#[cfg(test)]
mod tests {
    use crate::tests::check_diagnostics;

    #[test]
    fn async_blocks_are_borders() {
        check_diagnostics(
            r#"
fn foo() {
    'a: loop {
        async {
            break 'a;
          //^^^^^^^^ error: break outside of loop
               // ^^ error: use of unreachable label `'a`
            continue 'a;
          //^^^^^^^^^^^ error: continue outside of loop
                  // ^^ error: use of unreachable label `'a`
        };
    }
}
"#,
        );
    }

    #[test]
    fn closures_are_borders() {
        check_diagnostics(
            r#"
fn foo() {
    'a: loop {
        || {
            break 'a;
          //^^^^^^^^ error: break outside of loop
               // ^^ error: use of unreachable label `'a`
            continue 'a;
          //^^^^^^^^^^^ error: continue outside of loop
                  // ^^ error: use of unreachable label `'a`
        };
    }
}
"#,
        );
    }

    #[test]
    fn blocks_pass_through() {
        check_diagnostics(
            r#"
fn foo() {
    'a: loop {
        {
          break 'a;
          continue 'a;
        }
    }
}
"#,
        );
    }

    #[test]
    fn try_blocks_pass_through() {
        check_diagnostics(
            r#"
fn foo() {
    'a: loop {
        try {
            break 'a;
            continue 'a;
        };
    }
}
"#,
        );
    }
}
