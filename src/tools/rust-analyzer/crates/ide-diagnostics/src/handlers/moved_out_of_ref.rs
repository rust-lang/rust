use crate::{Diagnostic, DiagnosticsContext};
use hir::HirDisplay;

// Diagnostic: moved-out-of-ref
//
// This diagnostic is triggered on moving non copy things out of references.
pub(crate) fn moved_out_of_ref(ctx: &DiagnosticsContext<'_>, d: &hir::MovedOutOfRef) -> Diagnostic {
    Diagnostic::new(
        "moved-out-of-ref",
        format!("cannot move `{}` out of reference", d.ty.display(ctx.sema.db)),
        ctx.sema.diagnostics_display_range(d.span.clone()).range,
    )
    .experimental() // spans are broken, and I'm not sure how precise we can detect copy types
}

#[cfg(test)]
mod tests {
    use crate::tests::check_diagnostics;

    // FIXME: spans are broken

    #[test]
    fn move_by_explicit_deref() {
        check_diagnostics(
            r#"
struct X;
fn main() {
    let a = &X;
    let b = *a;
      //^ error: cannot move `X` out of reference
}
"#,
        );
    }

    #[test]
    fn move_out_of_field() {
        check_diagnostics(
            r#"
//- minicore: copy
struct X;
struct Y(X, i32);
fn main() {
    let a = &Y(X, 5);
    let b = a.0;
      //^ error: cannot move `X` out of reference
    let y = a.1;
}
"#,
        );
    }

    #[test]
    fn move_out_of_static() {
        check_diagnostics(
            r#"
//- minicore: copy
struct X;
fn main() {
    static S: X = X;
    let s = S;
      //^ error: cannot move `X` out of reference
}
"#,
        );
    }

    #[test]
    fn generic_types() {
        check_diagnostics(
            r#"
//- minicore: derive, copy

#[derive(Copy)]
struct X<T>(T);
struct Y;

fn consume<T>(_: X<T>) {

}

fn main() {
    let a = &X(Y);
    consume(*a);
  //^^^^^^^^^^^ error: cannot move `X<Y>` out of reference
    let a = &X(5);
    consume(*a);
}
"#,
        );
    }

    #[test]
    fn no_false_positive_simple() {
        check_diagnostics(
            r#"
//- minicore: copy
fn f(_: i32) {}
fn main() {
    let x = &2;
    f(*x);
}
"#,
        );
    }

    #[test]
    fn no_false_positive_unknown_type() {
        check_diagnostics(
            r#"
//- minicore: derive, copy
fn f(x: &Unknown) -> Unknown {
    *x
}

#[derive(Copy)]
struct X<T>(T);

struct Y<T>(T);

fn g(x: &X<Unknown>) -> X<Unknown> {
    *x
}

fn h(x: &Y<Unknown>) -> Y<Unknown> {
    // FIXME: we should show error for this, as `Y` is not copy
    // regardless of its generic parameter.
    *x
}

"#,
        );
    }

    #[test]
    fn no_false_positive_dyn_fn() {
        check_diagnostics(
            r#"
//- minicore: copy, fn
fn f(x: &mut &mut dyn Fn()) {
    x();
}

struct X<'a> {
    field: &'a mut dyn Fn(),
}

fn f(x: &mut X<'_>) {
    (x.field)();
}
"#,
        );
    }

    #[test]
    fn no_false_positive_match_and_closure_capture() {
        check_diagnostics(
            r#"
//- minicore: copy, fn
enum X {
    Foo(u16),
    Bar,
}

fn main() {
    let x = &X::Bar;
    let c = || match *x {
        X::Foo(t) => t,
        _ => 5,
    };
}
            "#,
        );
    }
}
