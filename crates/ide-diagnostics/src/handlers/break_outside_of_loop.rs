use crate::{Diagnostic, DiagnosticsContext};

// Diagnostic: break-outside-of-loop
//
// This diagnostic is triggered if the `break` keyword is used outside of a loop.
pub(crate) fn break_outside_of_loop(
    ctx: &DiagnosticsContext<'_>,
    d: &hir::BreakOutsideOfLoop,
) -> Diagnostic {
    let construct = if d.is_break { "break" } else { "continue" };
    Diagnostic::new(
        "break-outside-of-loop",
        format!("{construct} outside of loop"),
        ctx.sema.diagnostics_display_range(d.expr.clone().map(|it| it.into())).range,
    )
}

#[cfg(test)]
mod tests {
    use crate::tests::check_diagnostics;

    #[test]
    fn outside_of_loop() {
        check_diagnostics(
            r#"
fn foo() {
    break;
  //^^^^^ error: break outside of loop
    break 'a;
  //^^^^^^^^ error: break outside of loop
    continue;
  //^^^^^^^^ error: continue outside of loop
    continue 'a;
  //^^^^^^^^^^^ error: continue outside of loop
}
"#,
        );
    }

    #[test]
    fn try_blocks_are_borders() {
        check_diagnostics(
            r#"
fn foo() {
    'a: loop {
        try {
                break;
              //^^^^^ error: break outside of loop
                break 'a;
              //^^^^^^^^ error: break outside of loop
                continue;
              //^^^^^^^^ error: continue outside of loop
                continue 'a;
              //^^^^^^^^^^^ error: continue outside of loop
        };
    }
}
"#,
        );
    }

    #[test]
    fn async_blocks_are_borders() {
        check_diagnostics(
            r#"
fn foo() {
    'a: loop {
        try {
                break;
              //^^^^^ error: break outside of loop
                break 'a;
              //^^^^^^^^ error: break outside of loop
                continue;
              //^^^^^^^^ error: continue outside of loop
                continue 'a;
              //^^^^^^^^^^^ error: continue outside of loop
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
        try {
                break;
              //^^^^^ error: break outside of loop
                break 'a;
              //^^^^^^^^ error: break outside of loop
                continue;
              //^^^^^^^^ error: continue outside of loop
                continue 'a;
              //^^^^^^^^^^^ error: continue outside of loop
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
            break;
            break 'a;
            continue;
            continue 'a;
        }
    }
}
"#,
        );
    }

    #[test]
    fn label_blocks() {
        check_diagnostics(
            r#"
fn foo() {
    'a: {
        break;
      //^^^^^ error: break outside of loop
        break 'a;
        continue;
      //^^^^^^^^ error: continue outside of loop
        continue 'a;
      //^^^^^^^^^^^ error: continue outside of loop
    }
}
"#,
        );
    }
}
