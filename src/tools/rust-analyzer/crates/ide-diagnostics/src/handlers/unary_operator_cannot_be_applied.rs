use hir::HirDisplay;
use syntax::ast::UnaryOp;

use crate::{Diagnostic, DiagnosticCode, DiagnosticsContext};

// Diagnostic: unary-operator-cannot-be-applied
//
// This diagnostic is triggered if a unary operator (`!` or `-`) is applied
// to a value whose type does not implement the corresponding trait
// (`Not` or `Neg`).
pub(crate) fn unary_operator_cannot_be_applied(
    ctx: &DiagnosticsContext<'_, '_>,
    d: &hir::UnaryOperatorCannotBeApplied<'_>,
) -> Diagnostic {
    let op = match d.op {
        UnaryOp::Not => "!",
        UnaryOp::Neg => "-",
        // `Deref` uses a different diagnostic (`CannotBeDereferenced`).
        UnaryOp::Deref => "*",
    };
    Diagnostic::new_with_syntax_node_ptr(
        ctx,
        DiagnosticCode::RustcHardError("E0600"),
        format!(
            "cannot apply unary operator `{op}` to type `{}`",
            d.found.display(ctx.sema.db, ctx.display_target)
        ),
        d.expr.map(Into::into),
    )
    .stable()
}

#[cfg(test)]
mod tests {
    use crate::tests::check_diagnostics;

    #[test]
    fn not_on_enum() {
        check_diagnostics(
            r#"
//- minicore: unary_ops, builtin_impls
enum Question { Yes, No }

fn f() {
    let _ = !Question::Yes;
          //^^^^^^^^^^^^^^ error: cannot apply unary operator `!` to type `Question`
}
"#,
        );
    }

    #[test]
    fn neg_on_struct() {
        check_diagnostics(
            r#"
//- minicore: unary_ops, builtin_impls
struct S;

fn f() {
    let _ = -S;
          //^^ error: cannot apply unary operator `-` to type `S`
}
"#,
        );
    }

    #[test]
    fn allows_not_on_bool() {
        check_diagnostics(
            r#"
//- minicore: unary_ops, builtin_impls
fn f() {
    let _ = !true;
    let _ = !false;
}
"#,
        );
    }

    #[test]
    fn allows_not_on_integer() {
        check_diagnostics(
            r#"
//- minicore: unary_ops, builtin_impls
fn f() {
    let _ = !0u32;
    let _ = !0i32;
}
"#,
        );
    }

    #[test]
    fn allows_neg_on_numeric() {
        check_diagnostics(
            r#"
//- minicore: unary_ops, builtin_impls
fn f() {
    let _ = -1i32;
    let _ = -1.0f64;
}
"#,
        );
    }

    #[test]
    fn neg_on_unsigned() {
        check_diagnostics(
            r#"
//- minicore: unary_ops, builtin_impls
fn f() {
    let _ = -1u32;
          //^^^^^ error: cannot apply unary operator `-` to type `u32`
}
"#,
        );
    }

    #[test]
    fn allows_not_with_impl() {
        check_diagnostics(
            r#"
//- minicore: unary_ops, builtin_impls
struct Bar;
struct Foo;

impl core::ops::Not for Bar {
    type Output = Foo;
    fn not(self) -> Foo { Foo }
}

fn f() {
    let _ = !Bar;
}
"#,
        );
    }

    #[test]
    fn allows_neg_with_impl() {
        check_diagnostics(
            r#"
//- minicore: unary_ops, builtin_impls
struct Bar;
struct Foo;

impl core::ops::Neg for Bar {
    type Output = Foo;
    fn neg(self) -> Foo { Foo }
}

fn f() {
    let _ = -Bar;
}
"#,
        );
    }
}
