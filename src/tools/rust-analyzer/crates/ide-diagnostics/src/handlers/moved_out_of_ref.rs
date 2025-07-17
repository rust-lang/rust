use crate::{Diagnostic, DiagnosticCode, DiagnosticsContext};
use hir::HirDisplay;

// Diagnostic: moved-out-of-ref
//
// This diagnostic is triggered on moving non copy things out of references.
pub(crate) fn moved_out_of_ref(
    ctx: &DiagnosticsContext<'_>,
    d: &hir::MovedOutOfRef<'_>,
) -> Diagnostic {
    Diagnostic::new_with_syntax_node_ptr(
        ctx,
        DiagnosticCode::RustcHardError("E0507"),
        format!("cannot move `{}` out of reference", d.ty.display(ctx.sema.db, ctx.display_target)),
        d.span,
    )
    // spans are broken, and I'm not sure how precise we can detect copy types
}

#[cfg(test)]
mod tests {
    use crate::tests::check_diagnostics;

    #[test]
    fn operand_field_span_respected() {
        check_diagnostics(
            r#"
struct NotCopy;
struct S {
    field: NotCopy,
}

fn f(s: &S) -> S {
    S { field: s.field }
             //^^^^^^^ error: cannot move `NotCopy` out of reference
}
            "#,
        );
    }

    #[test]
    fn move_by_explicit_deref() {
        check_diagnostics(
            r#"
struct X;
fn main() {
    let a = &X;
    let b = *a;
      //^ error: cannot move `X` out of reference
    _ = b;
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
    _ = (b, y);
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
    let _s = S;
      //^^ error: cannot move `X` out of reference
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
          //^^ error: cannot move `X<Y>` out of reference
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
    let _c = || match *x {
        X::Foo(t) => t,
        _ => 5,
    };
}
            "#,
        );
    }

    #[test]
    fn regression_15787() {
        check_diagnostics(
            r#"
//- minicore: coerce_unsized, slice, copy
fn foo(mut slice: &[u32]) -> usize {
    slice = match slice {
        [0, rest @ ..] | rest => rest,
    };
    slice.len()
}
"#,
        );
    }

    #[test]
    fn regression_16564() {
        check_diagnostics(
            r#"
//- minicore: copy
fn test() {
    let _x = (&(&mut (),)).0 as *const ();
}
            "#,
        )
    }

    #[test]
    fn regression_18201() {
        check_diagnostics(
            r#"
//- minicore: copy
struct NotCopy;
struct S(NotCopy);
impl S {
    fn f(&mut self) {
        || {
            if let ref mut _cb = self.0 {
            }
        };
    }
}
"#,
        )
    }

    #[test]
    fn regression_20155() {
        check_diagnostics(
            r#"
//- minicore: copy, option
struct Box(i32);
fn test() {
    let b = Some(Box(0));
    || {
        if let Some(b) = b {
            let _move = b;
        }
    };
}
"#,
        )
    }
}
