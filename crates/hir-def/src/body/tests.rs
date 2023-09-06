mod block;

use base_db::{fixture::WithFixture, SourceDatabase};
use expect_test::{expect, Expect};

use crate::{test_db::TestDB, ModuleDefId};

use super::*;

fn lower(ra_fixture: &str) -> (TestDB, Arc<Body>, DefWithBodyId) {
    let db = TestDB::with_files(ra_fixture);

    let krate = db.crate_graph().iter().next().unwrap();
    let def_map = db.crate_def_map(krate);
    let mut fn_def = None;
    'outer: for (_, module) in def_map.modules() {
        for decl in module.scope.declarations() {
            if let ModuleDefId::FunctionId(it) = decl {
                fn_def = Some(it);
                break 'outer;
            }
        }
    }
    let fn_def = fn_def.unwrap().into();

    let body = db.body(fn_def);
    (db, body, fn_def)
}

fn def_map_at(ra_fixture: &str) -> String {
    let (db, position) = TestDB::with_position(ra_fixture);

    let module = db.module_at_position(position);
    module.def_map(&db).dump(&db)
}

fn check_block_scopes_at(ra_fixture: &str, expect: Expect) {
    let (db, position) = TestDB::with_position(ra_fixture);

    let module = db.module_at_position(position);
    let actual = module.def_map(&db).dump_block_scopes(&db);
    expect.assert_eq(&actual);
}

fn check_at(ra_fixture: &str, expect: Expect) {
    let actual = def_map_at(ra_fixture);
    expect.assert_eq(&actual);
}

#[test]
fn your_stack_belongs_to_me() {
    cov_mark::check!(your_stack_belongs_to_me);
    lower(
        r#"
macro_rules! n_nuple {
    ($e:tt) => ();
    ($($rest:tt)*) => {{
        (n_nuple!($($rest)*)None,)
    }};
}
fn main() { n_nuple!(1,2,3); }
"#,
    );
}

#[test]
fn your_stack_belongs_to_me2() {
    cov_mark::check!(overflow_but_not_me);
    lower(
        r#"
macro_rules! foo {
    () => {{ foo!(); foo!(); }}
}
fn main() { foo!(); }
"#,
    );
}

#[test]
fn recursion_limit() {
    cov_mark::check!(your_stack_belongs_to_me);

    lower(
        r#"
#![recursion_limit = "2"]
macro_rules! n_nuple {
    ($e:tt) => ();
    ($first:tt $($rest:tt)*) => {{
        n_nuple!($($rest)*)
    }};
}
fn main() { n_nuple!(1,2,3); }
"#,
    );
}

#[test]
fn issue_3642_bad_macro_stackover() {
    lower(
        r#"
#[macro_export]
macro_rules! match_ast {
    (match $node:ident { $($tt:tt)* }) => { match_ast!(match ($node) { $($tt)* }) };

    (match ($node:expr) {
        $( ast::$ast:ident($it:ident) => $res:expr, )*
        _ => $catch_all:expr $(,)?
    }) => {{
        $( if let Some($it) = ast::$ast::cast($node.clone()) { $res } else )*
        { $catch_all }
    }};
}

fn main() {
    let anchor = match_ast! {
        match parent {
            as => {},
            _ => return None
        }
    };
}"#,
    );
}

#[test]
fn macro_resolve() {
    // Regression test for a path resolution bug introduced with inner item handling.
    lower(
        r#"
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
"#,
    );
}

#[test]
fn desugar_builtin_format_args() {
    // Regression test for a path resolution bug introduced with inner item handling.
    let (db, body, def) = lower(
        r#"
//- minicore: fmt
fn main() {
    let are = "are";
    let count = 10;
    builtin#format_args("hello {count:02} {} friends, we {are:?} {0}{last}", "fancy", last = "!");
}
"#,
    );

    expect![[r#"
        fn main() {
            let are = "are";
            let count = 10;
            builtin#lang(Arguments::new_v1_formatted)(
                &[
                    "\"hello ", " ", " friends, we ", " ", "", "\"",
                ],
                &[
                    builtin#lang(Argument::new_display)(
                        &count,
                    ), builtin#lang(Argument::new_display)(
                        &"fancy",
                    ), builtin#lang(Argument::new_debug)(
                        &are,
                    ), builtin#lang(Argument::new_display)(
                        &"!",
                    ),
                ],
                &[
                    builtin#lang(Placeholder::new)(
                        0usize,
                        ' ',
                        builtin#lang(Alignment::Unknown),
                        8u32,
                        builtin#lang(Count::Implied),
                        builtin#lang(Count::Is)(
                            2usize,
                        ),
                    ), builtin#lang(Placeholder::new)(
                        1usize,
                        ' ',
                        builtin#lang(Alignment::Unknown),
                        0u32,
                        builtin#lang(Count::Implied),
                        builtin#lang(Count::Implied),
                    ), builtin#lang(Placeholder::new)(
                        2usize,
                        ' ',
                        builtin#lang(Alignment::Unknown),
                        0u32,
                        builtin#lang(Count::Implied),
                        builtin#lang(Count::Implied),
                    ), builtin#lang(Placeholder::new)(
                        1usize,
                        ' ',
                        builtin#lang(Alignment::Unknown),
                        0u32,
                        builtin#lang(Count::Implied),
                        builtin#lang(Count::Implied),
                    ), builtin#lang(Placeholder::new)(
                        3usize,
                        ' ',
                        builtin#lang(Alignment::Unknown),
                        0u32,
                        builtin#lang(Count::Implied),
                        builtin#lang(Count::Implied),
                    ),
                ],
                unsafe {
                    builtin#lang(UnsafeArg::new)()
                },
            );
        }"#]]
    .assert_eq(&body.pretty_print(&db, def))
}
