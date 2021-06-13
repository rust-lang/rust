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
