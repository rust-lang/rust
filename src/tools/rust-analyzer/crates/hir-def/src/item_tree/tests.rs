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

            // AstId: ExternCrate[5A82, 0]
            pub(self) extern crate self as renamed;

            // AstId: ExternCrate[7E1C, 0]
            pub(in super) extern crate bli;

            // AstId: Use[0000, 0]
            pub use crate::path::{nested, items as renamed, Trait as _};

            // AstId: Use[0000, 1]
            pub(self) use globs::*;

            #[doc = " docs on import"]
            // AstId: Use[0000, 2]
            pub(self) use crate::{A, B};

            // AstId: Use[0000, 3]
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
        expect![[r#"
            #[on_extern_block]
            // AstId: ExternBlock[0000, 0]
            extern {
                #[on_extern_type]
                // AstId: TypeAlias[9FDF, 0]
                pub(self) type ExType;

                #[on_extern_static]
                // AstId: Static[43C1, 0]
                pub(self) static EX_STATIC = _;

                #[on_extern_fn]
                // AstId: Fn[452D, 0]
                pub(self) fn ex_fn;
            }
        "#]],
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
            // AstId: Struct[DFF3, 0]
            pub(self) struct Unit;

            #[derive(Debug)]
            // AstId: Struct[C7A1, 0]
            pub(self) struct Struct { ... }

            // AstId: Struct[DAC2, 0]
            pub(self) struct Tuple(...);

            // AstId: Union[2DBB, 0]
            pub(self) union Ize { ... }

            // AstId: Enum[7FF8, 0]
            pub(self) enum E { ... }
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
            // AstId: Static[B393, 0]
            pub static ST = _;

            // AstId: Const[B309, 0]
            pub(self) const _ = _;

            #[attr]
            #[inner_attr_in_fn]
            // AstId: Fn[75E3, 0]
            pub(self) fn f;

            // AstId: Trait[2998, 0]
            pub(self) trait Tr { ... }
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
            // AstId: Module[CF93, 0]
            pub(self) mod inline {
                // AstId: Use[0000, 0]
                pub(self) use super::*;

                // AstId: Fn[1B26, 0]
                pub(self) fn fn_in_module;
            }

            // AstId: Module[8994, 0]
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
            // AstId: MacroRules[88CE, 0]
            macro_rules! m { ... }

            // AstId: MacroDef[DC34, 0]
            pub macro m2 { ... }

            // AstId: MacroCall[612F, 0], SyntaxContextId: ROOT2024, ExpandTo: Items
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
            // AstId: Struct[42E2, 0]
            pub(self) struct S;
        "#]],
    )
}
