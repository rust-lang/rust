mod block;

use base_db::{fixture::WithFixture, SourceDatabase};
use expect_test::Expect;

use crate::{test_db::TestDB, ModuleDefId};

use super::*;

fn lower(ra_fixture: &str) -> Arc<Body> {
    let db = crate::test_db::TestDB::with_files(ra_fixture);

    let krate = db.crate_graph().iter().next().unwrap();
    let def_map = db.crate_def_map(krate);
    let mut fn_def = None;
    'outer: for (_, module) in def_map.modules() {
        for decl in module.scope.declarations() {
            match decl {
                ModuleDefId::FunctionId(it) => {
                    fn_def = Some(it);
                    break 'outer;
                }
                _ => {}
            }
        }
    }

    db.body(fn_def.unwrap().into())
}

fn check_diagnostics(ra_fixture: &str) {
    let db: TestDB = TestDB::with_files(ra_fixture);
    db.check_diagnostics();
}

fn block_def_map_at(ra_fixture: &str) -> String {
    let (db, position) = crate::test_db::TestDB::with_position(ra_fixture);

    let module = db.module_at_position(position);
    module.def_map(&db).dump(&db)
}

fn check_at(ra_fixture: &str, expect: Expect) {
    let actual = block_def_map_at(ra_fixture);
    expect.assert_eq(&actual);
}

#[test]
fn your_stack_belongs_to_me() {
    cov_mark::check!(your_stack_belongs_to_me);
    lower(
        "
macro_rules! n_nuple {
    ($e:tt) => ();
    ($($rest:tt)*) => {{
        (n_nuple!($($rest)*)None,)
    }};
}
fn main() { n_nuple!(1,2,3); }
",
    );
}

#[test]
fn macro_resolve() {
    // Regression test for a path resolution bug introduced with inner item handling.
    lower(
        r"
macro_rules! vec {
    () => { () };
    ($elem:expr; $n:expr) => { () };
    ($($x:expr),+ $(,)?) => { () };
}
mod m {
    fn outer() {
        let _ = vec![FileSet::default(); self.len()];
    }
}
      ",
    );
}

#[test]
fn cfg_diagnostics() {
    check_diagnostics(
        r"
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
    ",
    );
}

#[test]
fn macro_diag_builtin() {
    check_diagnostics(
        r#"
#[rustc_builtin_macro]
macro_rules! env {}

#[rustc_builtin_macro]
macro_rules! include {}

#[rustc_builtin_macro]
macro_rules! compile_error {}

#[rustc_builtin_macro]
macro_rules! format_args {
    () => {}
}

fn f() {
    // Test a handful of built-in (eager) macros:

    include!(invalid);
  //^^^^^^^^^^^^^^^^^ could not convert tokens
    include!("does not exist");
  //^^^^^^^^^^^^^^^^^^^^^^^^^^ could not convert tokens

    env!(invalid);
  //^^^^^^^^^^^^^ could not convert tokens

    env!("OUT_DIR");
  //^^^^^^^^^^^^^^^ `OUT_DIR` not set, enable "load out dirs from check" to fix

    compile_error!("compile_error works");
  //^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ compile_error works

    // Lazy:

    format_args!();
  //^^^^^^^^^^^^^^ no rule matches input tokens
}
        "#,
    );
}

#[test]
fn macro_rules_diag() {
    check_diagnostics(
        r#"
macro_rules! m {
    () => {};
}
fn f() {
    m!();

    m!(hi);
  //^^^^^^ leftover tokens
}
      "#,
    );
}

#[test]
fn dollar_crate_in_builtin_macro() {
    check_diagnostics(
        r#"
#[macro_export]
#[rustc_builtin_macro]
macro_rules! format_args {}

#[macro_export]
macro_rules! arg {
    () => {}
}

#[macro_export]
macro_rules! outer {
    () => {
        $crate::format_args!( "", $crate::arg!(1) )
    };
}

fn f() {
    outer!();
  //^^^^^^^^ leftover tokens
}
        "#,
    )
}
