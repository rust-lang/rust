use crate::{Diagnostic, DiagnosticCode, DiagnosticsContext};
use hir::GenericArgKind;
use syntax::SyntaxKind;

// Diagnostic: incorrect-generics-order
//
// This diagnostic is triggered the order of provided generic arguments does not match their declaration.
pub(crate) fn incorrect_generics_order(
    ctx: &DiagnosticsContext<'_>,
    d: &hir::IncorrectGenericsOrder,
) -> Diagnostic {
    let provided_description = match d.provided_arg.value.kind() {
        SyntaxKind::CONST_ARG => "constant",
        SyntaxKind::LIFETIME_ARG => "lifetime",
        SyntaxKind::TYPE_ARG => "type",
        _ => panic!("non-generic-arg passed to `incorrect_generics_order()`"),
    };
    let expected_description = match d.expected_kind {
        GenericArgKind::Lifetime => "lifetime",
        GenericArgKind::Type => "type",
        GenericArgKind::Const => "constant",
    };
    let message =
        format!("{provided_description} provided when a {expected_description} was expected");
    Diagnostic::new_with_syntax_node_ptr(
        ctx,
        DiagnosticCode::RustcHardError("E0747"),
        message,
        d.provided_arg.map(Into::into),
    )
    .stable()
}

#[cfg(test)]
mod tests {
    use crate::tests::check_diagnostics;

    #[test]
    fn lifetime_out_of_order() {
        check_diagnostics(
            r#"
struct Foo<'a, T>(&'a T);

fn bar(_v: Foo<(), 'static>) {}
            // ^^ error: type provided when a lifetime was expected
        "#,
        );
    }

    #[test]
    fn types_and_consts() {
        check_diagnostics(
            r#"
struct Foo<T>(T);
fn foo1(_v: Foo<1>) {}
             // ^ error: constant provided when a type was expected
fn foo2(_v: Foo<{ (1, 2) }>) {}
             // ^^^^^^^^^^ error: constant provided when a type was expected

struct Bar<const N: usize>;
fn bar(_v: Bar<()>) {}
            // ^^ error: type provided when a constant was expected

struct Baz<T, const N: usize>(T);
fn baz(_v: Baz<1, ()>) {}
            // ^ error: constant provided when a type was expected
        "#,
        );
    }

    #[test]
    fn no_error_when_num_incorrect() {
        check_diagnostics(
            r#"
struct Baz<T, U>(T, U);
fn baz(_v: Baz<1>) {}
           // ^^^ error: this struct takes 2 generic arguments but 1 generic argument was supplied
        "#,
        );
    }
}
