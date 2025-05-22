use crate::{Diagnostic, DiagnosticCode, DiagnosticsContext};

// Diagnostic: missing-lifetime
//
// This diagnostic is triggered when a lifetime argument is missing.
pub(crate) fn missing_lifetime(
    ctx: &DiagnosticsContext<'_>,
    d: &hir::MissingLifetime,
) -> Diagnostic {
    Diagnostic::new_with_syntax_node_ptr(
        ctx,
        DiagnosticCode::RustcHardError("E0106"),
        "missing lifetime specifier",
        d.generics_or_segment.map(Into::into),
    )
}

#[cfg(test)]
mod tests {
    use crate::tests::check_diagnostics;

    #[test]
    fn in_fields() {
        check_diagnostics(
            r#"
struct Foo<'a>(&'a ());
struct Bar(Foo);
        // ^^^ error: missing lifetime specifier
        "#,
        );
    }

    #[test]
    fn bounds() {
        check_diagnostics(
            r#"
struct Foo<'a, T>(&'a T);
trait Trait<'a> {
    type Assoc;
}

fn foo<'a, T: Trait>(
           // ^^^^^ error: missing lifetime specifier
    _: impl Trait<'a, Assoc: Trait>,
                          // ^^^^^ error: missing lifetime specifier
)
where
    Foo<T>: Trait<'a>,
    // ^^^ error: missing lifetime specifier
{
}
        "#,
        );
    }

    #[test]
    fn generic_defaults() {
        check_diagnostics(
            r#"
struct Foo<'a>(&'a ());

struct Bar<T = Foo>(T);
            // ^^^ error: missing lifetime specifier
        "#,
        );
    }

    #[test]
    fn type_alias_type() {
        check_diagnostics(
            r#"
struct Foo<'a>(&'a ());

type Bar = Foo;
        // ^^^ error: missing lifetime specifier
        "#,
        );
    }

    #[test]
    fn const_param_ty() {
        check_diagnostics(
            r#"
struct Foo<'a>(&'a ());

fn bar<const F: Foo>() {}
             // ^^^ error: missing lifetime specifier
        "#,
        );
    }
}
