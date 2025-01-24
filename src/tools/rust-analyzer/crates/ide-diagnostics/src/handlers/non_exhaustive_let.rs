use crate::{Diagnostic, DiagnosticCode, DiagnosticsContext};

// Diagnostic: non-exhaustive-let
//
// This diagnostic is triggered if a `let` statement without an `else` branch has a non-exhaustive
// pattern.
pub(crate) fn non_exhaustive_let(
    ctx: &DiagnosticsContext<'_>,
    d: &hir::NonExhaustiveLet,
) -> Diagnostic {
    Diagnostic::new_with_syntax_node_ptr(
        ctx,
        DiagnosticCode::RustcHardError("E0005"),
        format!("non-exhaustive pattern: {}", d.uncovered_patterns),
        d.pat.map(Into::into),
    )
}

#[cfg(test)]
mod tests {
    use crate::tests::check_diagnostics;

    #[test]
    fn option_nonexhaustive() {
        check_diagnostics(
            r#"
//- minicore: option
fn main() {
    let None = Some(5);
      //^^^^ error: non-exhaustive pattern: `Some(_)` not covered
}
"#,
        );
    }

    #[test]
    fn option_exhaustive() {
        check_diagnostics(
            r#"
//- minicore: option
fn main() {
    let Some(_) | None = Some(5);
}
"#,
        );
    }

    #[test]
    fn option_nonexhaustive_inside_blocks() {
        check_diagnostics(
            r#"
//- minicore: option
fn main() {
    '_a: {
        let None = Some(5);
          //^^^^ error: non-exhaustive pattern: `Some(_)` not covered
    }
}
"#,
        );

        check_diagnostics(
            r#"
//- minicore: future, option
fn main() {
    let _ = async {
        let None = Some(5);
          //^^^^ error: non-exhaustive pattern: `Some(_)` not covered
    };
}
"#,
        );

        check_diagnostics(
            r#"
//- minicore: option
fn main() {
    unsafe {
        let None = Some(5);
          //^^^^ error: non-exhaustive pattern: `Some(_)` not covered
    }
}
"#,
        );
    }

    #[test]
    fn min_exhaustive() {
        check_diagnostics(
            r#"
//- minicore: result
fn test(x: Result<i32, !>) {
    let Ok(_y) = x;
}
"#,
        );

        check_diagnostics(
            r#"
//- minicore: result
fn test(x: Result<i32, &'static !>) {
    let Ok(_y) = x;
      //^^^^^^ error: non-exhaustive pattern: `Err(_)` not covered
}
"#,
        );
    }
}
