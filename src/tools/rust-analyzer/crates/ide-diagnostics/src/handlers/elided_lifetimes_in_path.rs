use crate::{Diagnostic, DiagnosticCode, DiagnosticsContext};

// Diagnostic: elided-lifetimes-in-path
//
// This diagnostic is triggered when lifetimes are elided in paths. It is a lint only for some cases,
// and a hard error for others.
pub(crate) fn elided_lifetimes_in_path(
    ctx: &DiagnosticsContext<'_>,
    d: &hir::ElidedLifetimesInPath,
) -> Diagnostic {
    if d.hard_error {
        Diagnostic::new_with_syntax_node_ptr(
            ctx,
            DiagnosticCode::RustcHardError("E0726"),
            "implicit elided lifetime not allowed here",
            d.generics_or_segment.map(Into::into),
        )
    } else {
        Diagnostic::new_with_syntax_node_ptr(
            ctx,
            DiagnosticCode::RustcLint("elided_lifetimes_in_paths"),
            "hidden lifetime parameters in types are deprecated",
            d.generics_or_segment.map(Into::into),
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::tests::check_diagnostics;

    #[test]
    fn fn_() {
        check_diagnostics(
            r#"
#![warn(elided_lifetimes_in_paths)]

struct Foo<'a>(&'a ());

fn foo(_: Foo) {}
       // ^^^ warn: hidden lifetime parameters in types are deprecated
        "#,
        );
        check_diagnostics(
            r#"
#![warn(elided_lifetimes_in_paths)]

struct Foo<'a>(&'a ());

fn foo(_: Foo<'_>) -> Foo { loop {} }
                   // ^^^ warn: hidden lifetime parameters in types are deprecated
        "#,
        );
    }

    #[test]
    fn async_fn() {
        check_diagnostics(
            r#"
struct Foo<'a>(&'a ());

async fn foo(_: Foo) {}
             // ^^^ error: implicit elided lifetime not allowed here
        "#,
        );
        check_diagnostics(
            r#"
#![warn(elided_lifetimes_in_paths)]

struct Foo<'a>(&'a ());

fn foo(_: Foo<'_>) -> Foo { loop {} }
                   // ^^^ warn: hidden lifetime parameters in types are deprecated
        "#,
        );
    }

    #[test]
    fn no_error_when_explicitly_elided() {
        check_diagnostics(
            r#"
#![warn(elided_lifetimes_in_paths)]

struct Foo<'a>(&'a ());
trait Trait<'a> {}

fn foo(_: Foo<'_>) -> Foo<'_> { loop {} }
async fn bar(_: Foo<'_>) -> Foo<'_> { loop {} }
impl Foo<'_> {}
impl Trait<'_> for Foo<'_> {}
        "#,
        );
    }

    #[test]
    fn impl_() {
        check_diagnostics(
            r#"
struct Foo<'a>(&'a ());
trait Trait<'a> {}

impl Foo {}
  // ^^^ error: implicit elided lifetime not allowed here

impl Trait for Foo<'_> {}
  // ^^^^^ error: implicit elided lifetime not allowed here
        "#,
        );
    }
}
