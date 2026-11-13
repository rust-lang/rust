use crate::{Diagnostic, DiagnosticCode, DiagnosticsContext};

// Diagnostic: method-call-illegal-sized-bound
//
// This diagnostic is triggered when a method is called on a trait-object
// receiver but the method's predicates require `Self: Sized`, which the
// trait object cannot satisfy.
pub(crate) fn method_call_illegal_sized_bound(
    ctx: &DiagnosticsContext<'_, '_>,
    d: &hir::MethodCallIllegalSizedBound,
) -> Diagnostic {
    Diagnostic::new_with_syntax_node_ptr(
        ctx,
        DiagnosticCode::RustcHardError("E0277"),
        "the method cannot be invoked on a trait object because its `Self: Sized` bound is not satisfied",
        d.call_expr.map(|it| it.into()),
    )
    .stable()
}

#[cfg(test)]
mod tests {
    use crate::tests::check_diagnostics;

    #[test]
    fn sized_bound_method_on_trait_object_errors() {
        check_diagnostics(
            r#"
//- minicore: sized
trait Foo {
    fn cant_call(&self) where Self: Sized;
}

fn f(x: &dyn Foo) {
    x.cant_call();
  //^^^^^^^^^^^^^ error: the method cannot be invoked on a trait object because its `Self: Sized` bound is not satisfied
}
"#,
        );
    }

    #[test]
    fn method_without_sized_bound_on_trait_object_does_not_error() {
        check_diagnostics(
            r#"
//- minicore: sized
trait Foo {
    fn dyn_safe(&self);
}

fn f(x: &dyn Foo) {
    x.dyn_safe();
}
"#,
        );
    }

    #[test]
    fn sized_bound_method_on_concrete_type_does_not_error() {
        check_diagnostics(
            r#"
//- minicore: sized
trait Foo {
    fn cant_dispatch(&self) where Self: Sized;
}

struct S;
impl Foo for S {
    fn cant_dispatch(&self) {}
}

fn f(s: &S) {
    s.cant_dispatch();
}
"#,
        );
    }
}
