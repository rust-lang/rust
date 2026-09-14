use crate::{Diagnostic, DiagnosticCode, DiagnosticsContext};

// Diagnostic: duplicate-field
//
// This diagnostic is triggered when a record expression or pattern specifies
// the same field more than once.
pub(crate) fn duplicate_field(
    ctx: &DiagnosticsContext<'_, '_>,
    d: &hir::DuplicateField,
) -> Diagnostic {
    Diagnostic::new_with_syntax_node_ptr(
        ctx,
        DiagnosticCode::RustcHardError("E0062"),
        "field specified more than once",
        d.field.map(Into::into),
    )
    .stable()
}

#[cfg(test)]
mod tests {
    use crate::tests::check_diagnostics;

    #[test]
    fn duplicate_field_in_struct_literal() {
        check_diagnostics(
            r#"
struct S { foo: i32, bar: i32 }
fn main() {
    let _ = S {
        foo: 1,
        bar: 2,
        foo: 3,
      //^^^^^^ error: field specified more than once
    };
}
"#,
        );
    }

    #[test]
    fn duplicate_field_in_enum_variant_literal() {
        check_diagnostics(
            r#"
enum E { V { foo: i32 } }
fn main() {
    let _ = E::V {
        foo: 1,
        foo: 2,
      //^^^^^^ error: field specified more than once
    };
}
"#,
        );
    }

    #[test]
    fn no_duplicate_when_each_field_specified_once() {
        check_diagnostics(
            r#"
struct S { foo: i32, bar: i32 }
fn main() {
    let _ = S { foo: 1, bar: 2 };
}
"#,
        );
    }

    #[test]
    fn no_duplicate_for_unknown_field_falls_through_to_no_such_field() {
        check_diagnostics(
            r#"
struct S { foo: i32 }
fn main() {
    let _ = S {
        foo: 1,
        bar: 2,
      //^^^^^^ 💡 error: no such field
    };
}
"#,
        );
    }

    #[test]
    fn duplicate_field_in_struct_pattern() {
        check_diagnostics(
            r#"
struct S { foo: i32, bar: i32 }
fn f(s: S) {
    let S {
        foo,
        bar,
        foo,
      //^^^ error: field specified more than once
        ..
    } = s;
    let _ = (foo, bar);
}
"#,
        );
    }

    #[test]
    fn duplicate_field_in_enum_variant_pattern() {
        check_diagnostics(
            r#"
enum E { V { foo: i32, bar: i32 } }
fn f(e: E) {
    match e {
        E::V {
            foo,
            bar,
            foo,
          //^^^ error: field specified more than once
            ..
        } => { let _ = (foo, bar); }
    }
}
"#,
        );
    }
}
