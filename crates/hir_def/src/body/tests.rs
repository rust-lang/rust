use base_db::{fixture::WithFixture, SourceDatabase};
use test_utils::mark;

use crate::{test_db::TestDB, ModuleDefId};

use super::*;

fn lower(ra_fixture: &str) -> Arc<Body> {
    let (db, file_id) = crate::test_db::TestDB::with_single_file(ra_fixture);

    let krate = db.crate_graph().iter().next().unwrap();
    let def_map = db.crate_def_map(krate);
    let module = def_map.modules_for_file(file_id).next().unwrap();
    let module = &def_map[module];
    let fn_def = match module.scope.declarations().next().unwrap() {
        ModuleDefId::FunctionId(it) => it,
        _ => panic!(),
    };

    db.body(fn_def.into())
}

fn check_diagnostics(ra_fixture: &str) {
    let db: TestDB = TestDB::with_files(ra_fixture);
    db.check_diagnostics();
}

#[test]
fn your_stack_belongs_to_me() {
    mark::check!(your_stack_belongs_to_me);
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
