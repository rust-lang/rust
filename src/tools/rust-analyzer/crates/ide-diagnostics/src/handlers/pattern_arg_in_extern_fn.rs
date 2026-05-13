use crate::{Diagnostic, DiagnosticCode, DiagnosticsContext};

// Diagnostic: pattern-arg-in-extern-fn
//
// This diagnostic is triggered if a pattern was declared as an argument in a foreign function declaration.
pub(crate) fn pattern_arg_in_extern_fn(
    ctx: &DiagnosticsContext<'_, '_>,
    d: &hir::PatternArgInExternFn,
) -> Diagnostic {
    Diagnostic::new_with_syntax_node_ptr(
        ctx,
        DiagnosticCode::RustcHardError("E0130"),
        "patterns aren't allowed in foreign function declarations",
        d.node.map(Into::into),
    )
    .stable()
}

#[cfg(test)]
mod tests {
    use crate::tests::check_diagnostics;

    #[test]
    fn tuple_pattern() {
        check_diagnostics(
            r#"
unsafe extern { fn foo((a, b): (u32, u32)); }
                    // ^^^^^^ error: patterns aren't allowed in foreign function declarations
            "#,
        );
    }

    #[test]
    fn struct_pattern() {
        check_diagnostics(
            r#"
struct Foo(u32, u32);
unsafe extern { fn foo(Foo(a, b): Foo); }
                    // ^^^^^^^^^ error: patterns aren't allowed in foreign function declarations
            "#,
        );

        check_diagnostics(
            r#"
struct Foo{ bar: u32, baz: u32 }
unsafe extern { fn foo(Foo { bar, baz }: Foo); }
                    // ^^^^^^^^^^^^^^^^ error: patterns aren't allowed in foreign function declarations
            "#,
        );
    }

    #[test]
    fn pattern_is_second_arg() {
        check_diagnostics(
            r#"
struct Foo(u32, u32);
unsafe extern { fn foo(okay: u32, Foo(a, b): Foo); }
                               // ^^^^^^^^^ error: patterns aren't allowed in foreign function declarations
            "#,
        );
    }

    #[test]
    fn non_simple_ident() {
        check_diagnostics(
            r#"
unsafe extern { fn foo(ref a: u32); }
                    // ^^^^^ error: patterns aren't allowed in foreign function declarations
            "#,
        );

        check_diagnostics(
            r#"
unsafe extern { fn foo(mut a: u32); }
                    // ^^^^^ error: patterns aren't allowed in foreign function declarations
            "#,
        );

        check_diagnostics(
            r#"
unsafe extern { fn foo(a @ _: u32); }
                    // ^^^^^ error: patterns aren't allowed in foreign function declarations
            "#,
        );
    }
}
