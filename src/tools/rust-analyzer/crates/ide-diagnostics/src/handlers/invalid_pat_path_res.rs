use hir::InvalidPatPathResKind;

use crate::{Diagnostic, DiagnosticCode, DiagnosticsContext};

// Diagnostic: invalid-pat-path-res
//
// triggered when a path pattern resolves to an item in the
// value namespace that cannot be matched in a pattern.
pub(crate) fn invalid_pat_path_res(
    ctx: &DiagnosticsContext<'_, '_>,
    d: &hir::InvalidPatPathRes,
) -> Diagnostic {
    let (code, message) = match d.kind {
        InvalidPatPathResKind::Static => (
            DiagnosticCode::RustcHardError("E0158"),
            "statics cannot be referenced in patterns".to_owned(),
        ),
        InvalidPatPathResKind::GenericParam => (
            DiagnosticCode::RustcHardError("E0158"),
            "expected unit struct, unit variant or constant, found generic parameter".to_owned(),
        ),
        InvalidPatPathResKind::Function => (
            DiagnosticCode::RustcHardError("E0532"),
            "expected unit struct, unit variant or constant, found function".to_owned(),
        ),
        InvalidPatPathResKind::ImplSelf => (
            DiagnosticCode::RustcHardError("E0532"),
            "expected unit struct, unit variant or constant, found `Self`".to_owned(),
        ),
        InvalidPatPathResKind::LocalBinding => (
            DiagnosticCode::RustcHardError("E0532"),
            "expected unit struct, unit variant or constant, found local variable".to_owned(),
        ),
    };

    Diagnostic::new_with_syntax_node_ptr(ctx, code, message, d.pat.map(Into::into)).stable()
}

#[cfg(test)]
mod tests {
    use crate::tests::check_diagnostics;

    #[test]
    fn static_in_path_pattern() {
        check_diagnostics(
            r#"
static S: i32 = 67;

fn main() {
    match 0 {
        self::S => {}
      //^^^^^^^ error: statics cannot be referenced in patterns
        _ => {}
    }
}
"#,
        );
    }

    #[test]
    fn fn_in_path_pattern() {
        check_diagnostics(
            r#"
mod m {
    pub fn foo() {}
}

struct S;
impl S {
    fn assoc_fn() {}
}

fn main() {
    match () {
        m::foo => {}
      //^^^^^^ error: expected unit struct, unit variant or constant, found function
        S::assoc_fn => {}
      //^^^^^^^^^^^ error: expected unit struct, unit variant or constant, found function
    }
}
"#,
        );
    }

    #[test]
    fn valid_const_and_unit_struct_in_pattern() {
        check_diagnostics(
            r#"
const C: i32 = 1;
struct Unit;

fn main() {
    match 1 {
        C => {}
        _ => {}
    }
    let Unit = Unit;
}
"#,
        );
    }
}
