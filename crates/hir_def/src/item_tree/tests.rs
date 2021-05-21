use base_db::fixture::WithFixture;
use expect_test::{expect, Expect};

use crate::{db::DefDatabase, test_db::TestDB};

fn check(ra_fixture: &str, expect: Expect) {
    let (db, file_id) = TestDB::with_single_file(ra_fixture);
    let item_tree = db.file_item_tree(file_id.into());
    let pretty = item_tree.pretty_print();
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
        "#,
        expect![[r##"
            #![doc = " file comment"]  // AttrId { is_doc_comment: true, ast_index: 0 }
            #![no_std]  // AttrId { is_doc_comment: false, ast_index: 0 }
            #![doc = " another file comment"]  // AttrId { is_doc_comment: true, ast_index: 1 }

            pub(self) extern crate self as renamed;

            pub(super) extern crate bli;

            pub use crate::path::nested;  // 0

            pub use crate::path::items as renamed;  // 1

            pub use crate::path::Trait as _;  // 2

            pub(self) use globs::*;  // 0

            #[doc = " docs on import"]  // AttrId { is_doc_comment: true, ast_index: 0 }
            pub(self) use crate::A;  // 0

            #[doc = " docs on import"]  // AttrId { is_doc_comment: true, ast_index: 0 }
            pub(self) use crate::B;  // 1
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
            #[on_extern_block]  // AttrId { is_doc_comment: false, ast_index: 0 }
            extern "C" {
                #[on_extern_type]  // AttrId { is_doc_comment: false, ast_index: 0 }
                pub(self) type ExType;  // extern

                #[on_extern_static]  // AttrId { is_doc_comment: false, ast_index: 0 }
                pub(self) static EX_STATIC: u8 = _;  // extern

                #[on_extern_fn]  // AttrId { is_doc_comment: false, ast_index: 0 }
                // flags = 0x60
                pub(self) fn ex_fn() -> ();
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
        expect![[r##"
            pub(self) struct Unit;

            #[derive(Debug)]  // AttrId { is_doc_comment: false, ast_index: 0 }
            pub(self) struct Struct {
                #[doc = " fld docs"]  // AttrId { is_doc_comment: true, ast_index: 0 }
                pub(self) fld: (),
            }

            pub(self) struct Tuple(
                #[attr]  // AttrId { is_doc_comment: false, ast_index: 0 }
                pub(self) 0: u8,
            );

            pub(self) union Ize {
                pub(self) a: (),
                pub(self) b: (),
            }

            pub(self) enum E {
                #[doc = " comment on Unit"]  // AttrId { is_doc_comment: true, ast_index: 0 }
                Unit,
                #[doc = " comment on Tuple"]  // AttrId { is_doc_comment: true, ast_index: 0 }
                Tuple(
                    pub(self) 0: u8,
                ),
                Struct {
                    #[doc = " comment on a: u8"]  // AttrId { is_doc_comment: true, ast_index: 0 }
                    pub(self) a: u8,
                },
            }
        "##]],
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
        expect![[r##"
            pub static mut ST: () = _;

            pub(self) const _: Anon = _;

            #[attr]  // AttrId { is_doc_comment: false, ast_index: 0 }
            #[inner_attr_in_fn]  // AttrId { is_doc_comment: false, ast_index: 1 }
            // flags = 0x2
            pub(self) fn f(
                #[attr]  // AttrId { is_doc_comment: false, ast_index: 0 }
                _: u8,
                _: (),
            ) -> ();

            pub(self) trait Tr: SuperTrait + 'lifetime {
                pub(self) type Assoc: AssocBound = Default;

                // flags = 0x1
                pub(self) fn method(
                    _: &Self,
                ) -> ();
            }
        "##]],
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
        "#,
        expect![[r##"
            #[doc = " outer"]  // AttrId { is_doc_comment: true, ast_index: 0 }
            #[doc = " inner"]  // AttrId { is_doc_comment: true, ast_index: 1 }
            pub(self) mod inline {
                pub(self) use super::*;  // 0

                // flags = 0x2
                pub(self) fn fn_in_module() -> ();
            }
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
            macro_rules! m { ... }

            pub macro m2 { ... }

            m!(...);
        "#]],
    );
}
