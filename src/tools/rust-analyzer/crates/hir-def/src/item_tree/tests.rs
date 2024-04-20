use expect_test::{expect, Expect};
use test_fixture::WithFixture;

use crate::{db::DefDatabase, test_db::TestDB};

fn check(ra_fixture: &str, expect: Expect) {
    let (db, file_id) = TestDB::with_single_file(ra_fixture);
    let item_tree = db.file_item_tree(file_id.into());
    let pretty = item_tree.pretty_print(&db);
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
                pub(self) static EX_STATIC: u8 = _;

                #[on_extern_fn]
                // AstId: 4
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
        expect![[r#"
            // AstId: 1
            pub(self) struct Unit;

            #[derive(Debug)]
            // AstId: 2
            pub(self) struct Struct {
                // AstId: 6
                #[doc = " fld docs"]
                pub(self) fld: (),
            }

            // AstId: 3
            pub(self) struct Tuple(
                // AstId: 7
                #[attr]
                pub(self) 0: u8,
            );

            // AstId: 4
            pub(self) union Ize {
                // AstId: 8
                pub(self) a: (),
                // AstId: 9
                pub(self) b: (),
            }

            // AstId: 5
            pub(self) enum E {
                // AstId: 10
                #[doc = " comment on Unit"]
                Unit,
                // AstId: 11
                #[doc = " comment on Tuple"]
                Tuple(
                    // AstId: 13
                    pub(self) 0: u8,
                ),
                // AstId: 12
                Struct {
                    // AstId: 14
                    #[doc = " comment on a: u8"]
                    pub(self) a: u8,
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
            pub static mut ST: () = _;

            // AstId: 2
            pub(self) const _: Anon = _;

            #[attr]
            #[inner_attr_in_fn]
            // AstId: 3
            pub(self) fn f(
                #[attr]
                // AstId: 5
                u8,
                // AstId: 6
                (),
            ) -> () { ... }

            // AstId: 4
            pub(self) trait Tr<Self>
            where
                Self: SuperTrait,
                Self: 'lifetime
            {
                // AstId: 8
                pub(self) type Assoc: AssocBound = Default;

                // AstId: 9
                pub(self) fn method(
                    // AstId: 10
                    self: &Self,
                ) -> ();
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
                pub(self) fn fn_in_module() -> () { ... }
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

            // AstId: 3, SyntaxContext: 0, ExpandTo: Items
            m!(...);
        "#]],
    );
}

#[test]
fn mod_paths() {
    check(
        r#"
struct S {
    a: self::Ty,
    b: super::SuperTy,
    c: super::super::SuperSuperTy,
    d: ::abs::Path,
    e: crate::Crate,
    f: plain::path::Ty,
}
        "#,
        expect![[r#"
            // AstId: 1
            pub(self) struct S {
                // AstId: 2
                pub(self) a: self::Ty,
                // AstId: 3
                pub(self) b: super::SuperTy,
                // AstId: 4
                pub(self) c: super::super::SuperSuperTy,
                // AstId: 5
                pub(self) d: ::abs::Path,
                // AstId: 6
                pub(self) e: crate::Crate,
                // AstId: 7
                pub(self) f: plain::path::Ty,
            }
        "#]],
    )
}

#[test]
fn types() {
    check(
        r#"
struct S {
    a: Mixed<'a, T, Item=(), OtherItem=u8>,
    b: <Fully as Qualified>::Syntax,
    c: <TypeAnchored>::Path::<'a>,
    d: dyn for<'a> Trait<'a>,
}
        "#,
        expect![[r#"
            // AstId: 1
            pub(self) struct S {
                // AstId: 2
                pub(self) a: Mixed::<'a, T, Item = (), OtherItem = u8>,
                // AstId: 3
                pub(self) b: Qualified::<Self=Fully>::Syntax,
                // AstId: 4
                pub(self) c: <TypeAnchored>::Path::<'a>,
                // AstId: 5
                pub(self) d: dyn for<'a> Trait::<'a>,
            }
        "#]],
    )
}

#[test]
fn generics() {
    check(
        r#"
struct S<'a, 'b: 'a, T: Copy + 'a + 'b, const K: u8 = 0> {
    field: &'a &'b T,
}

struct Tuple<T: Copy, U: ?Sized>(T, U);

impl<'a, 'b: 'a, T: Copy + 'a + 'b, const K: u8 = 0> S<'a, 'b, T, K> {
    fn f<G: 'a>(arg: impl Copy) -> impl Copy {}
}

enum Enum<'a, T, const U: u8> {}
union Union<'a, T, const U: u8> {}

trait Tr<'a, T: 'a>: Super where Self: for<'a> Tr<'a, T> {}
        "#,
        expect![[r#"
            // AstId: 1
            pub(self) struct S<'a, 'b, T, const K: u8>
            where
                T: Copy,
                T: 'a,
                T: 'b
            {
                // AstId: 8
                pub(self) field: &'a &'b T,
            }

            // AstId: 2
            pub(self) struct Tuple<T, U>(
                // AstId: 9
                pub(self) 0: T,
                // AstId: 10
                pub(self) 1: U,
            )
            where
                T: Copy,
                U: ?Sized;

            // AstId: 3
            impl<'a, 'b, T, const K: u8> S::<'a, 'b, T, K>
            where
                T: Copy,
                T: 'a,
                T: 'b
            {
                // AstId: 12
                pub(self) fn f<G>(
                    // AstId: 13
                    impl Copy,
                ) -> impl Copy
                where
                    G: 'a { ... }
            }

            // AstId: 4
            pub(self) enum Enum<'a, T, const U: u8> {
            }

            // AstId: 5
            pub(self) union Union<'a, T, const U: u8> {
            }

            // AstId: 6
            pub(self) trait Tr<'a, Self, T>
            where
                Self: Super,
                T: 'a,
                Self: for<'a> Tr::<'a, T>
            {
            }
        "#]],
    )
}

#[test]
fn generics_with_attributes() {
    check(
        r#"
struct S<#[cfg(never)] T>;
        "#,
        expect![[r#"
            // AstId: 1
            pub(self) struct S<#[cfg(never)] T>;
        "#]],
    )
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
