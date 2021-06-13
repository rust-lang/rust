use cfg::DnfExpr;
use stdx::format_to;

use crate::diagnostics::{Diagnostic, DiagnosticsContext};

// Diagnostic: inactive-code
//
// This diagnostic is shown for code with inactive `#[cfg]` attributes.
pub(super) fn inactive_code(
    ctx: &DiagnosticsContext<'_>,
    d: &hir::InactiveCode,
) -> Option<Diagnostic> {
    // If there's inactive code somewhere in a macro, don't propagate to the call-site.
    if d.node.file_id.expansion_info(ctx.sema.db).is_some() {
        return None;
    }

    let inactive = DnfExpr::new(d.cfg.clone()).why_inactive(&d.opts);
    let mut message = "code is inactive due to #[cfg] directives".to_string();

    if let Some(inactive) = inactive {
        format_to!(message, ": {}", inactive);
    }

    let res = Diagnostic::new(
        "inactive-code",
        message,
        ctx.sema.diagnostics_display_range(d.node.clone()).range,
    )
    .with_unused(true);
    Some(res)
}

#[cfg(test)]
mod tests {
    use crate::diagnostics::tests::check_diagnostics_with_inactive_code;

    #[test]
    fn cfg_diagnostics() {
        check_diagnostics_with_inactive_code(
            r#"
fn f() {
    // The three g̶e̶n̶d̶e̶r̶s̶ statements:

    #[cfg(a)] fn f() {}  // Item statement
  //^^^^^^^^^^^^^^^^^^^ code is inactive due to #[cfg] directives: a is disabled
    #[cfg(a)] {}         // Expression statement
  //^^^^^^^^^^^^ code is inactive due to #[cfg] directives: a is disabled
    #[cfg(a)] let x = 0; // let statement
  //^^^^^^^^^^^^^^^^^^^^ code is inactive due to #[cfg] directives: a is disabled

    abc(#[cfg(a)] 0);
      //^^^^^^^^^^^ code is inactive due to #[cfg] directives: a is disabled
    let x = Struct {
        #[cfg(a)] f: 0,
      //^^^^^^^^^^^^^^ code is inactive due to #[cfg] directives: a is disabled
    };
    match () {
        () => (),
        #[cfg(a)] () => (),
      //^^^^^^^^^^^^^^^^^^ code is inactive due to #[cfg] directives: a is disabled
    }

    #[cfg(a)] 0          // Trailing expression of block
  //^^^^^^^^^^^ code is inactive due to #[cfg] directives: a is disabled
}
        "#,
            true,
        );
    }

    #[test]
    fn inactive_item() {
        // Additional tests in `cfg` crate. This only tests disabled cfgs.

        check_diagnostics_with_inactive_code(
            r#"
    #[cfg(no)] pub fn f() {}
  //^^^^^^^^^^^^^^^^^^^^^^^^ code is inactive due to #[cfg] directives: no is disabled

    #[cfg(no)] #[cfg(no2)] mod m;
  //^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ code is inactive due to #[cfg] directives: no and no2 are disabled

    #[cfg(all(not(a), b))] enum E {}
  //^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ code is inactive due to #[cfg] directives: b is disabled

    #[cfg(feature = "std")] use std;
  //^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ code is inactive due to #[cfg] directives: feature = "std" is disabled
"#,
            true,
        );
    }

    /// Tests that `cfg` attributes behind `cfg_attr` is handled properly.
    #[test]
    fn inactive_via_cfg_attr() {
        cov_mark::check!(cfg_attr_active);
        check_diagnostics_with_inactive_code(
            r#"
    #[cfg_attr(not(never), cfg(no))] fn f() {}
  //^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ code is inactive due to #[cfg] directives: no is disabled

    #[cfg_attr(not(never), cfg(not(no)))] fn f() {}

    #[cfg_attr(never, cfg(no))] fn g() {}

    #[cfg_attr(not(never), inline, cfg(no))] fn h() {}
  //^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ code is inactive due to #[cfg] directives: no is disabled
"#,
            true,
        );
    }
}
