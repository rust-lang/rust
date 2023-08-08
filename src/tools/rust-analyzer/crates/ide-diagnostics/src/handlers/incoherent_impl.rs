use hir::InFile;

use crate::{Diagnostic, DiagnosticCode, DiagnosticsContext};

// Diagnostic: incoherent-impl
//
// This diagnostic is triggered if the targe type of an impl is from a foreign crate.
pub(crate) fn incoherent_impl(ctx: &DiagnosticsContext<'_>, d: &hir::IncoherentImpl) -> Diagnostic {
    Diagnostic::new_with_syntax_node_ptr(
        ctx,
        DiagnosticCode::RustcHardError("E0210"),
        format!("cannot define inherent `impl` for foreign type"),
        InFile::new(d.file_id, d.impl_.clone().into()),
    )
}

#[cfg(test)]
mod change_case {
    use crate::tests::check_diagnostics;

    #[test]
    fn primitive() {
        check_diagnostics(
            r#"
  impl bool {}
//^^^^^^^^^^^^ error: cannot define inherent `impl` for foreign type
"#,
        );
    }

    #[test]
    fn primitive_rustc_allow_incoherent_impl() {
        check_diagnostics(
            r#"
impl bool {
    #[rustc_allow_incoherent_impl]
    fn falsch(self) -> Self { false }
}
"#,
        );
    }

    #[test]
    fn rustc_allow_incoherent_impl() {
        check_diagnostics(
            r#"
//- /lib.rs crate:foo
#[rustc_has_incoherent_inherent_impls]
pub struct S;
//- /main.rs crate:main deps:foo
impl foo::S {
    #[rustc_allow_incoherent_impl]
    fn func(self) {}
}
"#,
        );
        check_diagnostics(
            r#"
//- /lib.rs crate:foo
pub struct S;
//- /main.rs crate:main deps:foo
  impl foo::S { #[rustc_allow_incoherent_impl] fn func(self) {} }
//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ error: cannot define inherent `impl` for foreign type
"#,
        );
        check_diagnostics(
            r#"
//- /lib.rs crate:foo
#[rustc_has_incoherent_inherent_impls]
pub struct S;
//- /main.rs crate:main deps:foo
  impl foo::S { fn func(self) {} }
//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ error: cannot define inherent `impl` for foreign type
"#,
        );
    }
}
