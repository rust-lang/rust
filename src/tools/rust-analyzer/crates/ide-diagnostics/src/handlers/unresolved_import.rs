use crate::{Diagnostic, DiagnosticCode, DiagnosticsContext};

// Diagnostic: unresolved-import
//
// This diagnostic is triggered if rust-analyzer is unable to resolve a path in
// a `use` declaration.
pub(crate) fn unresolved_import(
    ctx: &DiagnosticsContext<'_>,
    d: &hir::UnresolvedImport,
) -> Diagnostic {
    Diagnostic::new_with_syntax_node_ptr(
        ctx,
        DiagnosticCode::RustcHardError("E0432"),
        "unresolved import",
        d.decl.map(|it| it.into()),
    )
    // This currently results in false positives in the following cases:
    // - `cfg_if!`-generated code in libstd (we don't load the sysroot correctly)
    // - `core::arch` (we don't handle `#[path = "../<path>"]` correctly)
    // - proc macros and/or proc macro generated code
}

#[cfg(test)]
mod tests {
    use crate::tests::check_diagnostics;

    #[test]
    fn unresolved_import() {
        check_diagnostics(
            r#"
use does_exist;
use does_not_exist;
  //^^^^^^^^^^^^^^ error: unresolved import

mod does_exist {}
"#,
        );
    }

    #[test]
    fn unresolved_import_in_use_tree() {
        // Only the relevant part of a nested `use` item should be highlighted.
        check_diagnostics(
            r#"
use does_exist::{Exists, DoesntExist};
                       //^^^^^^^^^^^ error: unresolved import

use {does_not_exist::*, does_exist};
   //^^^^^^^^^^^^^^^^^ error: unresolved import

use does_not_exist::{
    a,
  //^ error: unresolved import
    b,
  //^ error: unresolved import
    c,
  //^ error: unresolved import
};

mod does_exist {
    pub struct Exists;
}
"#,
        );
    }

    #[test]
    fn dedup_unresolved_import_from_unresolved_crate() {
        check_diagnostics(
            r#"
//- /main.rs crate:main
mod a {
    extern crate doesnotexist;
  //^^^^^^^^^^^^^^^^^^^^^^^^^^ error: unresolved extern crate

    // Should not error, since we already errored for the missing crate.
    use doesnotexist::{self, bla, *};

    use crate::doesnotexist;
      //^^^^^^^^^^^^^^^^^^^ error: unresolved import
}

mod m {
    use super::doesnotexist;
      //^^^^^^^^^^^^^^^^^^^ error: unresolved import
}
"#,
        );
    }
}
