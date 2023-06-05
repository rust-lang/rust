use crate::{Diagnostic, DiagnosticsContext};

// Diagnostic: unreachable-label
pub(crate) fn unreachable_label(
    ctx: &DiagnosticsContext<'_>,
    d: &hir::UnreachableLabel,
) -> Diagnostic {
    let name = &d.name;
    Diagnostic::new(
        "unreachable-label",
        format!("use of unreachable label `{}`", name.display(ctx.sema.db)),
        ctx.sema.diagnostics_display_range(d.node.clone().map(|it| it.into())).range,
    )
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
