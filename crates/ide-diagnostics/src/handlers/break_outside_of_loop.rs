use crate::{Diagnostic, DiagnosticsContext};

// Diagnostic: break-outside-of-loop
//
// This diagnostic is triggered if the `break` keyword is used outside of a loop.
pub(crate) fn break_outside_of_loop(
    ctx: &DiagnosticsContext<'_>,
    d: &hir::BreakOutsideOfLoop,
) -> Diagnostic {
    let message = if d.bad_value_break {
        "can't break with a value in this position".to_owned()
    } else {
        let construct = if d.is_break { "break" } else { "continue" };
        format!("{construct} outside of loop")
    };
    Diagnostic::new(
        "break-outside-of-loop",
        message,
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
    continue;
  //^^^^^^^^ error: continue outside of loop
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
        async {
                break;
              //^^^^^ error: break outside of loop
                continue;
              //^^^^^^^^ error: continue outside of loop
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
                break;
              //^^^^^ error: break outside of loop
                continue;
              //^^^^^^^^ error: continue outside of loop
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
            continue;
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
                break;
                continue;
        };
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
        continue;
      //^^^^^^^^ error: continue outside of loop
    }
}
"#,
        );
    }

    #[test]
    fn value_break_in_for_loop() {
        // FIXME: the error is correct, but the message is terrible
        check_diagnostics(
            r#"
//- minicore: iterator
fn test() {
    for _ in [()] {
        break 3;
           // ^ error: expected (), found i32
    }
}
"#,
        );
    }

    #[test]
    fn try_block_desugaring_inside_closure() {
        // regression test for #14701
        check_diagnostics(
            r#"
//- minicore: option, try
fn test() {
    try {
        || {
            let x = Some(2);
            Some(x?)
        };
    };
}
"#,
        );
    }
}
