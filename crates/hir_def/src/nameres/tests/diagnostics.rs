use base_db::fixture::WithFixture;

use crate::test_db::TestDB;

fn check_diagnostics(ra_fixture: &str) {
    let db: TestDB = TestDB::with_files(ra_fixture);
    db.check_diagnostics();
}

fn check_no_diagnostics(ra_fixture: &str) {
    let db: TestDB = TestDB::with_files(ra_fixture);
    db.check_no_diagnostics();
}

#[test]
fn unresolved_import() {
    check_diagnostics(
        r"
        use does_exist;
        use does_not_exist;
      //^^^^^^^^^^^^^^^^^^^ UnresolvedImport

        mod does_exist {}
        ",
    );
}

#[test]
fn unresolved_extern_crate() {
    check_diagnostics(
        r"
        //- /main.rs crate:main deps:core
        extern crate core;
          extern crate doesnotexist;
        //^^^^^^^^^^^^^^^^^^^^^^^^^^ UnresolvedExternCrate
        //- /lib.rs crate:core
        ",
    );
}

#[test]
fn extern_crate_self_as() {
    cov_mark::check!(extern_crate_self_as);
    check_diagnostics(
        r"
        //- /lib.rs
          extern crate doesnotexist;
        //^^^^^^^^^^^^^^^^^^^^^^^^^^ UnresolvedExternCrate
        // Should not error.
        extern crate self as foo;
        struct Foo;
        use foo::Foo as Bar;
        ",
    );
}

#[test]
fn dedup_unresolved_import_from_unresolved_crate() {
    check_diagnostics(
        r"
        //- /main.rs crate:main
        mod a {
            extern crate doesnotexist;
          //^^^^^^^^^^^^^^^^^^^^^^^^^^ UnresolvedExternCrate

            // Should not error, since we already errored for the missing crate.
            use doesnotexist::{self, bla, *};

            use crate::doesnotexist;
          //^^^^^^^^^^^^^^^^^^^^^^^^ UnresolvedImport
        }

        mod m {
            use super::doesnotexist;
          //^^^^^^^^^^^^^^^^^^^^^^^^ UnresolvedImport
        }
        ",
    );
}

#[test]
fn unresolved_module() {
    check_diagnostics(
        r"
        //- /lib.rs
        mod foo;
          mod bar;
        //^^^^^^^^ UnresolvedModule
        mod baz {}
        //- /foo.rs
        ",
    );
}

#[test]
fn inactive_item() {
    // Additional tests in `cfg` crate. This only tests disabled cfgs.

    check_diagnostics(
        r#"
        //- /lib.rs
          #[cfg(no)] pub fn f() {}
        //^^^^^^^^^^^^^^^^^^^^^^^^ UnconfiguredCode

          #[cfg(no)] #[cfg(no2)] mod m;
        //^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ UnconfiguredCode

          #[cfg(all(not(a), b))] enum E {}
        //^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ UnconfiguredCode

          #[cfg(feature = "std")] use std;
        //^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ UnconfiguredCode
        "#,
    );
}

/// Tests that `cfg` attributes behind `cfg_attr` is handled properly.
#[test]
fn inactive_via_cfg_attr() {
    cov_mark::check!(cfg_attr_active);
    check_diagnostics(
        r#"
        //- /lib.rs
          #[cfg_attr(not(never), cfg(no))] fn f() {}
        //^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ UnconfiguredCode

          #[cfg_attr(not(never), cfg(not(no)))] fn f() {}

          #[cfg_attr(never, cfg(no))] fn g() {}

          #[cfg_attr(not(never), inline, cfg(no))] fn h() {}
        //^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ UnconfiguredCode
        "#,
    );
}

#[test]
fn unresolved_legacy_scope_macro() {
    check_diagnostics(
        r#"
        //- /lib.rs
          macro_rules! m { () => {} }

          m!();
          m2!();
        //^^^^^^ UnresolvedMacroCall
        "#,
    );
}

#[test]
fn unresolved_module_scope_macro() {
    check_diagnostics(
        r#"
        //- /lib.rs
          mod mac {
            #[macro_export]
            macro_rules! m { () => {} }
          }

          self::m!();
          self::m2!();
        //^^^^^^^^^^^^ UnresolvedMacroCall
        "#,
    );
}

#[test]
fn builtin_macro_fails_expansion() {
    check_diagnostics(
        r#"
        //- /lib.rs
          #[rustc_builtin_macro]
          macro_rules! include { () => {} }

          include!("doesntexist");
        //^^^^^^^^^^^^^^^^^^^^^^^^ failed to load file `doesntexist`
        "#,
    );
}

#[test]
fn include_macro_should_allow_empty_content() {
    check_no_diagnostics(
        r#"
        //- /lib.rs
          #[rustc_builtin_macro]
          macro_rules! include { () => {} }

          include!("bar.rs");
        //- /bar.rs
          // empty
        "#,
    );
}

#[test]
fn good_out_dir_diagnostic() {
    check_diagnostics(
        r#"
        #[rustc_builtin_macro]
        macro_rules! include { () => {} }
        #[rustc_builtin_macro]
        macro_rules! env { () => {} }
        #[rustc_builtin_macro]
        macro_rules! concat { () => {} }

        include!(concat!(env!("OUT_DIR"), "/out.rs"));
      //^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ `OUT_DIR` not set, enable "run build scripts" to fix
        "#,
    );
}

#[test]
fn register_attr_and_tool() {
    cov_mark::check!(register_attr);
    cov_mark::check!(register_tool);
    check_no_diagnostics(
        r#"
#![register_tool(tool)]
#![register_attr(attr)]

#[tool::path]
#[attr]
struct S;
        "#,
    );
    // NB: we don't currently emit diagnostics here
}
