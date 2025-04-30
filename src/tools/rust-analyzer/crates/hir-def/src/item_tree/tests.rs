use expect_test::{Expect, expect};
use span::Edition;
use test_fixture::WithFixture;

use crate::{db::DefDatabase, test_db::TestDB};

fn check(#[rust_analyzer::rust_fixture] ra_fixture: &str, expect: Expect) {
    let (db, file_id) = TestDB::with_single_file(ra_fixture);
    let item_tree = db.file_item_tree(file_id.into());
    let pretty = item_tree.pretty_print(&db, Edition::CURRENT);
    expect.assert_eq(&pretty);
}

#[test]
fn imports() {
    check(
        r#"
//! file comment
#![no_std]
//! another file comment

extern crate self as renamed;
pub(super) extern crate bli;

pub use crate::path::{nested, items as renamed, Trait as _};
use globs::*;

/// docs on import
use crate::{A, B};

use a::{c, d::{e}};
        "#,
        expect![[r##"
            #![doc = " file comment"]
            #![no_std]
            #![doc = " another file comment"]

            // AstId: 1
            pub(self) extern crate self as renamed;

            // AstId: 2
            pub(super) extern crate bli;

            // AstId: 3
            pub use crate::path::{nested, items as renamed, Trait as _};

            // AstId: 4
            pub(self) use globs::*;

            #[doc = " docs on import"]
            // AstId: 5
            pub(self) use crate::{A, B};

            // AstId: 6
            pub(self) use a::{c, d::{e}};
        "##]],
    );
}

#[test]
fn extern_blocks() {
    check(
        r#"
#[on_extern_block]
extern "C" {
    #[on_extern_type]
    type ExType;

    #[on_extern_static]
    static EX_STATIC: u8;

    #[on_extern_fn]
    fn ex_fn();
}
        "#,
        expect![[r##"
            #[on_extern_block]
            // AstId: 1
            extern "C" {
                #[on_extern_type]
                // AstId: 2
                pub(self) type ExType;

                #[on_extern_static]
                // AstId: 3
                pub(self) static EX_STATIC = _;

                #[on_extern_fn]
                // AstId: 4
                pub(self) fn ex_fn;
            }
        "##]],
    );
}

#[test]
fn adts() {
    check(
        r#"
struct Unit;

#[derive(Debug)]
struct Struct {
    /// fld docs
    fld: (),
}

struct Tuple(#[attr] u8);

union Ize {
    a: (),
    b: (),
}

enum E {
    /// comment on Unit
    Unit,
    /// comment on Tuple
    Tuple(u8),
    Struct {
        /// comment on a: u8
        a: u8,
    }
}
        "#,
        expect![[r#"
            // AstId: 1
            pub(self) struct Unit;

            #[derive(Debug)]
            // AstId: 2
            pub(self) struct Struct {
                #[doc = " fld docs"]
                pub(self) fld,
            }

            // AstId: 3
            pub(self) struct Tuple(
                #[attr]
                pub(self) 0,
            );

            // AstId: 4
            pub(self) union Ize {
                pub(self) a,
                pub(self) b,
            }

            // AstId: 5
            pub(self) enum E
                // AstId: 6
                #[doc = " comment on Unit"]
                Unit,
                // AstId: 7
                #[doc = " comment on Tuple"]
                Tuple(
                    pub(self) 0,
                ),
                // AstId: 8
                Struct {
                    #[doc = " comment on a: u8"]
                    pub(self) a,
                },
            }
        "#]],
    );
}

#[test]
fn misc() {
    check(
        r#"
pub static mut ST: () = ();

const _: Anon = ();

#[attr]
fn f(#[attr] arg: u8, _: ()) {
    #![inner_attr_in_fn]
}

trait Tr: SuperTrait + 'lifetime {
    type Assoc: AssocBound = Default;
    fn method(&self);
}
        "#,
        expect![[r#"
            // AstId: 1
            pub static ST = _;

            // AstId: 2
            pub(self) const _ = _;

            #[attr]
            #[inner_attr_in_fn]
            // AstId: 3
            pub(self) fn f;

            // AstId: 4
            pub(self) trait Tr {
                // AstId: 6
                pub(self) type Assoc;

                // AstId: 7
                pub(self) fn method;
            }
        "#]],
    );
}

#[test]
fn modules() {
    check(
        r#"
/// outer
mod inline {
    //! inner

    use super::*;

    fn fn_in_module() {}
}

mod outline;
        "#,
        expect![[r##"
            #[doc = " outer"]
            #[doc = " inner"]
            // AstId: 1
            pub(self) mod inline {
                // AstId: 3
                pub(self) use super::*;

                // AstId: 4
                pub(self) fn fn_in_module;
            }

            // AstId: 2
            pub(self) mod outline;
        "##]],
    );
}

#[test]
fn macros() {
    check(
        r#"
macro_rules! m {
    () => {};
}

pub macro m2() {}

m!();
        "#,
        expect![[r#"
            // AstId: 1
            macro_rules! m { ... }

            // AstId: 2
            pub macro m2 { ... }

            // AstId: 3, SyntaxContextId: ROOT2024, ExpandTo: Items
            m!(...);
        "#]],
    );
}

#[test]
fn pub_self() {
    check(
        r#"
pub(self) struct S;
        "#,
        expect![[r#"
            // AstId: 1
            pub(self) struct S;
        "#]],
    )
}
