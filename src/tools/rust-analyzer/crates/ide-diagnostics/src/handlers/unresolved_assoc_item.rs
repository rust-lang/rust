use crate::{Diagnostic, DiagnosticCode, DiagnosticsContext};

// Diagnostic: unresolved-assoc-item
//
// This diagnostic is triggered if the referenced associated item does not exist.
pub(crate) fn unresolved_assoc_item(
    ctx: &DiagnosticsContext<'_>,
    d: &hir::UnresolvedAssocItem,
) -> Diagnostic {
    Diagnostic::new_with_syntax_node_ptr(
        ctx,
        DiagnosticCode::RustcHardError("E0599"),
        "no such associated item",
        d.expr_or_pat.map(Into::into),
    )
}

#[cfg(test)]
mod tests {
    use crate::tests::check_diagnostics;

    #[test]
    fn bare() {
        check_diagnostics(
            r#"
struct S;

fn main() {
    let _ = S::Assoc;
          //^^^^^^^^ error: no such associated item
}
"#,
        );
    }

    #[test]
    fn unimplemented_trait() {
        check_diagnostics(
            r#"
struct S;
trait Foo {
    const X: u32;
}

fn main() {
    let _ = S::X;
          //^^^^ error: no such associated item
}
"#,
        );
    }

    #[test]
    fn dyn_super_trait_assoc_type() {
        check_diagnostics(
            r#"
//- minicore: future, send

use core::{future::Future, marker::Send, pin::Pin};

trait FusedFuture: Future {
    fn is_terminated(&self) -> bool;
}

struct Box<T: ?Sized>(*const T);

fn main() {
    let _fut: Pin<Box<dyn FusedFuture<Output = ()> + Send>> = loop {};
}
"#,
        );
    }
}
