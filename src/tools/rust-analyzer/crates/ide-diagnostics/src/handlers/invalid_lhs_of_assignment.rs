use crate::{Diagnostic, DiagnosticCode, DiagnosticsContext};

// Diagnostic: invalid-lhs-of-assignment
//
// This diagnostic is triggered if the left-hand side of an assignment can't be assigned to.
pub(crate) fn invalid_lhs_of_assignment(
    ctx: &DiagnosticsContext<'_, '_>,
    d: &hir::InvalidLhsOfAssignment,
) -> Diagnostic {
    Diagnostic::new_with_syntax_node_ptr(
        ctx,
        DiagnosticCode::RustcHardError("E0067"),
        "invalid left-hand side of assignment",
        d.lhs.map(Into::into),
    )
    .stable()
}

#[cfg(test)]
mod tests {
    use crate::tests::check_diagnostics;

    #[test]
    fn unit_struct_literal() {
        check_diagnostics(
            r#"
//- minicore: add
struct Struct;
impl core::ops::AddAssign for Struct {
    fn add_assign(&mut self, _other: Self) {}
}
fn test() {
    Struct += Struct;
 // ^^^^^^ error: invalid left-hand side of assignment
}
        "#,
        );
    }

    #[test]
    fn struct_literal() {
        check_diagnostics(
            r#"
//- minicore: add
struct Struct { foo: i32, bar: i32 }
impl core::ops::AddAssign for Struct {
    fn add_assign(&mut self, _other: Self) {}
}
fn test() {
    Struct { foo: 0, bar: 0 } += Struct { foo: 1, bar: 2 };
 // ^^^^^^^^^^^^^^^^^^^^^^^^^ error: invalid left-hand side of assignment
}
        "#,
        );
    }

    #[test]
    fn destructuring_assignment() {
        // no diagnostic, as `=` is not a _compound_ assignment
        check_diagnostics(
            r#"
//- minicore: add
struct Struct { foo: i32, bar: i32 }
impl core::ops::AddAssign for Struct {
    fn add_assign(&mut self, _other: Self) {}
}
fn test(mut foo: i32, mut bar: i32) {
    Struct { foo, bar } = Struct { foo: 1, bar: 2 };
}
        "#,
        );
    }

    #[test]
    fn destructuring_compound_assignment() {
        check_diagnostics(
            r#"
//- minicore: add
struct Struct { foo: i32, bar: i32 }
impl core::ops::AddAssign for Struct {
    fn add_assign(&mut self, _other: Self) {}
}
fn test(foo: i32, bar: i32) {
    Struct { foo, bar } += Struct { foo: 1, bar: 2 };
 // ^^^^^^^^^^^^^^^^^^^ error: invalid left-hand side of assignment
}
        "#,
        );
    }
}
