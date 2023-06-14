use cfg::DnfExpr;
use stdx::format_to;

use crate::{Diagnostic, DiagnosticCode, DiagnosticsContext, Severity};

// Diagnostic: inactive-code
//
// This diagnostic is shown for code with inactive `#[cfg]` attributes.
pub(crate) fn inactive_code(
    ctx: &DiagnosticsContext<'_>,
    d: &hir::InactiveCode,
) -> Option<Diagnostic> {
    // If there's inactive code somewhere in a macro, don't propagate to the call-site.
    if d.node.file_id.is_macro() {
        return None;
    }

    let inactive = DnfExpr::new(d.cfg.clone()).why_inactive(&d.opts);
    let mut message = "code is inactive due to #[cfg] directives".to_string();

    if let Some(inactive) = inactive {
        let inactive_reasons = inactive.to_string();

        if inactive_reasons.is_empty() {
            format_to!(message);
        } else {
            format_to!(message, ": {}", inactive);
        }
    }
    // FIXME: This shouldn't be a diagnostic
    let res = Diagnostic::new(
        DiagnosticCode::Ra("inactive-code", Severity::WeakWarning),
        message,
        ctx.sema.diagnostics_display_range(d.node.clone()).range,
    )
    .with_unused(true);
    Some(res)
}

#[cfg(test)]
mod tests {
    use crate::{tests::check_diagnostics_with_config, DiagnosticsConfig};

    pub(crate) fn check(ra_fixture: &str) {
        let config = DiagnosticsConfig::test_sample();
        check_diagnostics_with_config(config, ra_fixture)
    }

    #[test]
    fn cfg_diagnostics() {
        check(
            r#"
fn f() {
    // The three g̶e̶n̶d̶e̶r̶s̶ statements:

    #[cfg(a)] fn f() {}  // Item statement
  //^^^^^^^^^^^^^^^^^^^ weak: code is inactive due to #[cfg] directives: a is disabled
    #[cfg(a)] {}         // Expression statement
  //^^^^^^^^^^^^ weak: code is inactive due to #[cfg] directives: a is disabled
    #[cfg(a)] let x = 0; // let statement
  //^^^^^^^^^^^^^^^^^^^^ weak: code is inactive due to #[cfg] directives: a is disabled

    abc(#[cfg(a)] 0);
      //^^^^^^^^^^^ weak: code is inactive due to #[cfg] directives: a is disabled
    let x = Struct {
        #[cfg(a)] f: 0,
      //^^^^^^^^^^^^^^ weak: code is inactive due to #[cfg] directives: a is disabled
    };
    match () {
        () => (),
        #[cfg(a)] () => (),
      //^^^^^^^^^^^^^^^^^^^ weak: code is inactive due to #[cfg] directives: a is disabled
    }

    #[cfg(a)] 0          // Trailing expression of block
  //^^^^^^^^^^^ weak: code is inactive due to #[cfg] directives: a is disabled
}
        "#,
        );
    }

    #[test]
    fn inactive_item() {
        // Additional tests in `cfg` crate. This only tests disabled cfgs.

        check(
            r#"
    #[cfg(no)] pub fn f() {}
  //^^^^^^^^^^^^^^^^^^^^^^^^ weak: code is inactive due to #[cfg] directives: no is disabled

    #[cfg(no)] #[cfg(no2)] mod m;
  //^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ weak: code is inactive due to #[cfg] directives: no and no2 are disabled

    #[cfg(all(not(a), b))] enum E {}
  //^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ weak: code is inactive due to #[cfg] directives: b is disabled

    #[cfg(feature = "std")] use std;
  //^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ weak: code is inactive due to #[cfg] directives: feature = "std" is disabled

    #[cfg(any())] pub fn f() {}
  //^^^^^^^^^^^^^^^^^^^^^^^^^^^ weak: code is inactive due to #[cfg] directives
"#,
        );
    }

    #[test]
    fn inactive_assoc_item() {
        check(
            r#"
struct Foo;
impl Foo {
    #[cfg(any())] pub fn f() {}
  //^^^^^^^^^^^^^^^^^^^^^^^^^^^ weak: code is inactive due to #[cfg] directives
}

trait Bar {
    #[cfg(any())] pub fn f() {}
  //^^^^^^^^^^^^^^^^^^^^^^^^^^^ weak: code is inactive due to #[cfg] directives
}
"#,
        );
    }

    /// Tests that `cfg` attributes behind `cfg_attr` is handled properly.
    #[test]
    fn inactive_via_cfg_attr() {
        cov_mark::check!(cfg_attr_active);
        check(
            r#"
    #[cfg_attr(not(never), cfg(no))] fn f() {}
  //^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ weak: code is inactive due to #[cfg] directives: no is disabled

    #[cfg_attr(not(never), cfg(not(no)))] fn f() {}

    #[cfg_attr(never, cfg(no))] fn g() {}

    #[cfg_attr(not(never), inline, cfg(no))] fn h() {}
  //^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ weak: code is inactive due to #[cfg] directives: no is disabled
"#,
        );
    }

    #[test]
    fn inactive_fields_and_variants() {
        check(
            r#"
enum Foo {
  #[cfg(a)] Bar,
//^^^^^^^^^^^^^ weak: code is inactive due to #[cfg] directives: a is disabled
  Baz {
    #[cfg(a)] baz: String,
  //^^^^^^^^^^^^^^^^^^^^^ weak: code is inactive due to #[cfg] directives: a is disabled
  },
  Qux(#[cfg(a)] String),
    //^^^^^^^^^^^^^^^^ weak: code is inactive due to #[cfg] directives: a is disabled
}

struct Baz {
  #[cfg(a)] baz: String,
//^^^^^^^^^^^^^^^^^^^^^ weak: code is inactive due to #[cfg] directives: a is disabled
}

struct Qux(#[cfg(a)] String);
         //^^^^^^^^^^^^^^^^ weak: code is inactive due to #[cfg] directives: a is disabled

union FooBar {
  #[cfg(a)] baz: u32,
//^^^^^^^^^^^^^^^^^^ weak: code is inactive due to #[cfg] directives: a is disabled
}
"#,
        );
    }
}
