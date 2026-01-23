mod globs;
mod imports;
mod incremental;
mod macros;
mod mod_resolution;

use base_db::RootQueryDb;
use expect_test::{Expect, expect};
use test_fixture::WithFixture;

use crate::{
    nameres::{DefMap, crate_def_map},
    test_db::TestDB,
};

fn compute_crate_def_map(
    #[rust_analyzer::rust_fixture] ra_fixture: &str,
    cb: impl FnOnce(&DefMap),
) {
    let db = TestDB::with_files(ra_fixture);
    let krate = db.fetch_test_crate();
    cb(crate_def_map(&db, krate));
}

fn render_crate_def_map(#[rust_analyzer::rust_fixture] ra_fixture: &str) -> String {
    let db = TestDB::with_files(ra_fixture);
    let krate = db.fetch_test_crate();
    crate_def_map(&db, krate).dump(&db)
}

fn check(#[rust_analyzer::rust_fixture] ra_fixture: &str, expect: Expect) {
    let actual = render_crate_def_map(ra_fixture);
    expect.assert_eq(&actual);
}

#[test]
fn crate_def_map_smoke_test() {
    check(
        r#"
//- /lib.rs
mod foo;
struct S;
use crate::foo::bar::E;
use self::E::V;

//- /foo/mod.rs
pub mod bar;
fn f() {}

//- /foo/bar.rs
pub struct Baz;

union U { to_be: bool, not_to_be: u8 }
enum E { V }

extern {
    type Ext;
    static EXT: u8;
    fn ext();
}
"#,
        expect![[r#"
            crate
            - E : _
            - S : type value
            - V : _
            - foo : type

            crate::foo
            - bar : type
            - f : value

            crate::foo::bar
            - Baz : type value
            - E : type
            - EXT : value
            - Ext : type
            - U : type
            - ext : value
        "#]],
    );
}

#[test]
fn crate_def_map_super_super() {
    check(
        r#"
mod a {
    const A: usize = 0;
    mod b {
        const B: usize = 0;
        mod c {
            use super::super::*;
        }
    }
}
"#,
        expect![[r#"
            crate
            - a : type

            crate::a
            - A : value
            - b : type

            crate::a::b
            - B : value
            - c : type

            crate::a::b::c
            - A : value (glob)
            - b : type (glob)
        "#]],
    );
}

#[test]
fn crate_def_map_fn_mod_same_name() {
    check(
        r#"
mod m {
    pub mod z {}
    pub fn z() {}
}
"#,
        expect![[r#"
            crate
            - m : type

            crate::m
            - z : type value

            crate::m::z
        "#]],
    );
}

#[test]
fn bogus_paths() {
    cov_mark::check!(bogus_paths);
    check(
        r#"
//- /lib.rs
mod foo;
struct S;
use self;

//- /foo/mod.rs
use super;
use crate;
"#,
        expect![[r#"
            crate
            - S : type value
            - foo : type

            crate::foo
        "#]],
    );
}

#[test]
fn use_as() {
    check(
        r#"
//- /lib.rs
mod foo;
use crate::foo::Baz as Foo;

//- /foo/mod.rs
pub struct Baz;
"#,
        expect![[r#"
            crate
            - Foo : type (import) value (import)
            - foo : type

            crate::foo
            - Baz : type value
        "#]],
    );
}

#[test]
fn use_trees() {
    check(
        r#"
//- /lib.rs
mod foo;
use crate::foo::bar::{Baz, Quux};

//- /foo/mod.rs
pub mod bar;

//- /foo/bar.rs
pub struct Baz;
pub enum Quux {};
"#,
        expect![[r#"
            crate
            - Baz : type (import) value (import)
            - Quux : type (import)
            - foo : type

            crate::foo
            - bar : type

            crate::foo::bar
            - Baz : type value
            - Quux : type
        "#]],
    );
}

#[test]
fn re_exports() {
    check(
        r#"
//- /lib.rs
mod foo;
use self::foo::Baz;

//- /foo/mod.rs
pub mod bar;
pub use self::bar::Baz;

//- /foo/bar.rs
pub struct Baz;
"#,
        expect![[r#"
            crate
            - Baz : type (import) value (import)
            - foo : type

            crate::foo
            - Baz : type (import) value (import)
            - bar : type

            crate::foo::bar
            - Baz : type value
        "#]],
    );
}

#[test]
fn std_prelude() {
    cov_mark::check!(std_prelude);
    check(
        r#"
//- /main.rs crate:main deps:test_crate
#[prelude_import]
use ::test_crate::prelude::*;

use Foo::*;

//- /lib.rs crate:test_crate
pub mod prelude;

//- /prelude.rs
pub enum Foo { Bar, Baz }
"#,
        expect![[r#"
            crate
            - Bar : type (glob) value (glob)
            - Baz : type (glob) value (glob)
        "#]],
    );
}

#[test]
fn can_import_enum_variant() {
    cov_mark::check!(can_import_enum_variant);
    check(
        r#"
enum E { V }
use self::E::V;
"#,
        expect![[r#"
            crate
            - E : type
            - V : type (import) value (import)
        "#]],
    );
}

#[test]
fn edition_2015_imports() {
    check(
        r#"
//- /main.rs crate:main deps:other_crate edition:2015
mod foo;
mod bar;

//- /bar.rs
struct Bar;

//- /foo.rs
use bar::Bar;
use other_crate::FromLib;

//- /lib.rs crate:other_crate edition:2018
pub struct FromLib;
"#,
        expect![[r#"
            crate
            - bar : type
            - foo : type

            crate::bar
            - Bar : type value

            crate::foo
            - Bar : _
            - FromLib : type (import) value (import)
        "#]],
    );
}

#[test]
fn item_map_using_self() {
    check(
        r#"
//- /lib.rs
mod foo;
use crate::foo::bar::Baz::{self};

//- /foo/mod.rs
pub mod bar;

//- /foo/bar.rs
pub struct Baz;
"#,
        expect![[r#"
            crate
            - Baz : type (import)
            - foo : type

            crate::foo
            - bar : type

            crate::foo::bar
            - Baz : type value
        "#]],
    );
}

#[test]
fn item_map_across_crates() {
    check(
        r#"
//- /main.rs crate:main deps:test_crate
use test_crate::Baz;

//- /lib.rs crate:test_crate
pub struct Baz;
"#,
        expect![[r#"
            crate
            - Baz : type (import) value (import)
        "#]],
    );
}

#[test]
fn extern_crate_rename() {
    check(
        r#"
//- /main.rs crate:main deps:alloc
extern crate alloc as alloc_crate;
mod alloc;
mod sync;

//- /sync.rs
use alloc_crate::Arc;

//- /lib.rs crate:alloc
pub struct Arc;
"#,
        expect![[r#"
            crate
            - alloc : type
            - alloc_crate : type (extern)
            - sync : type

            crate::alloc

            crate::sync
            - Arc : type (import) value (import)
        "#]],
    );
}

#[test]
fn extern_crate_reexport() {
    check(
        r#"
//- /main.rs crate:main deps:importer
use importer::*;
use importer::extern_crate1::exported::*;
use importer::allowed_reexport::*;
use importer::extern_crate2::*;
use importer::not_allowed_reexport1;
use importer::not_allowed_reexport2;

//- /importer.rs crate:importer deps:extern_crate1,extern_crate2
extern crate extern_crate1;
extern crate extern_crate2;

pub use extern_crate1;
pub use extern_crate1 as allowed_reexport;

pub use ::extern_crate;
pub use self::extern_crate as not_allowed_reexport1;
pub use crate::extern_crate as not_allowed_reexport2;

//- /extern_crate1.rs crate:extern_crate1
pub mod exported {
    pub struct PublicItem;
    struct PrivateItem;
}

pub struct Exported;

//- /extern_crate2.rs crate:extern_crate2
pub struct NotExported;
"#,
        expect![[r#"
            crate
            - Exported : type (glob) value (glob)
            - PublicItem : type (glob) value (glob)
            - allowed_reexport : type (glob)
            - exported : type (glob)
            - not_allowed_reexport1 : _
            - not_allowed_reexport2 : _
        "#]],
    );
}

#[test]
fn extern_crate_rename_2015_edition() {
    check(
        r#"
//- /main.rs crate:main deps:alloc edition:2015
extern crate alloc as alloc_crate;
mod alloc;
mod sync;

//- /sync.rs
use alloc_crate::Arc;

//- /lib.rs crate:alloc
pub struct Arc;
"#,
        expect![[r#"
            crate
            - alloc : type
            - alloc_crate : type (extern)
            - sync : type

            crate::alloc

            crate::sync
            - Arc : type (import) value (import)
        "#]],
    );
}

#[test]
fn macro_use_extern_crate_self() {
    check(
        r#"
//- /main.rs crate:main
#[macro_use]
extern crate self as bla;
"#,
        expect![[r#"
            crate
            - bla : type (extern)
        "#]],
    );
}

#[test]
fn reexport_across_crates() {
    check(
        r#"
//- /main.rs crate:main deps:test_crate
use test_crate::Baz;

//- /lib.rs crate:test_crate
pub use foo::Baz;
mod foo;

//- /foo.rs
pub struct Baz;
"#,
        expect![[r#"
            crate
            - Baz : type (import) value (import)
        "#]],
    );
}

#[test]
fn values_dont_shadow_extern_crates() {
    check(
        r#"
//- /main.rs crate:main deps:foo
fn foo() {}
use foo::Bar;

//- /foo/lib.rs crate:foo
pub struct Bar;
"#,
        expect![[r#"
            crate
            - Bar : type (import) value (import)
            - foo : value
        "#]],
    );
}

#[test]
fn no_std_prelude() {
    check(
        r#"
        //- /main.rs edition:2018 crate:main deps:core,std
        #![cfg_attr(not(never), no_std)]
        use Rust;

        //- /core.rs crate:core
        pub mod prelude {
            pub mod rust_2018 {
                pub struct Rust;
            }
        }
        //- /std.rs crate:std deps:core
        pub mod prelude {
            pub mod rust_2018 {
            }
        }
    "#,
        expect![[r#"
            crate
            - Rust : type (import) value (import)
        "#]],
    );
}

#[test]
fn edition_specific_preludes() {
    // We can't test the 2015 prelude here since you can't reexport its contents with 2015's
    // absolute paths.

    check(
        r#"
        //- /main.rs edition:2018 crate:main deps:std
        use Rust2018;

        //- /std.rs crate:std
        pub mod prelude {
            pub mod rust_2018 {
                pub struct Rust2018;
            }
        }
    "#,
        expect![[r#"
            crate
            - Rust2018 : type (import) value (import)
        "#]],
    );
    check(
        r#"
        //- /main.rs edition:2021 crate:main deps:std
        use Rust2021;

        //- /std.rs crate:std
        pub mod prelude {
            pub mod rust_2021 {
                pub struct Rust2021;
            }
        }
    "#,
        expect![[r#"
            crate
            - Rust2021 : type (import) value (import)
        "#]],
    );
}

#[test]
fn std_prelude_takes_precedence_above_core_prelude() {
    check(
        r#"
//- /main.rs edition:2018 crate:main deps:core,std
use {Foo, Bar};

//- /std.rs crate:std deps:core
pub mod prelude {
    pub mod rust_2018 {
        pub struct Foo;
        pub use core::prelude::rust_2018::Bar;
    }
}

//- /core.rs crate:core
pub mod prelude {
    pub mod rust_2018 {
        pub struct Bar;
    }
}
"#,
        expect![[r#"
            crate
            - Bar : type (import) value (import)
            - Foo : type (import) value (import)
        "#]],
    );
}

#[test]
fn cfg_not_test() {
    check(
        r#"
//- /main.rs edition:2018 crate:main deps:std
use {Foo, Bar, Baz};

//- /lib.rs crate:std
pub mod prelude {
    pub mod rust_2018 {
        #[cfg(test)]
        pub struct Foo;
        #[cfg(not(test))]
        pub struct Bar;
        #[cfg(all(not(any()), feature = "foo", feature = "bar", opt = "42"))]
        pub struct Baz;
    }
}
"#,
        expect![[r#"
            crate
            - Bar : type (import) value (import)
            - Baz : _
            - Foo : _
        "#]],
    );
}

#[test]
fn cfg_test() {
    check(
        r#"
//- /main.rs edition:2018 crate:main deps:std
use {Foo, Bar, Baz};

//- /lib.rs crate:std cfg:test,feature=foo,feature=bar,opt=42
pub mod prelude {
    pub mod rust_2018 {
        #[cfg(test)]
        pub struct Foo;
        #[cfg(not(test))]
        pub struct Bar;
        #[cfg(all(not(any()), feature = "foo", feature = "bar", opt = "42"))]
        pub struct Baz;
    }
}
"#,
        expect![[r#"
            crate
            - Bar : _
            - Baz : type (import) value (import)
            - Foo : type (import) value (import)
        "#]],
    );
}

#[test]
fn infer_multiple_namespace() {
    check(
        r#"
//- /main.rs
mod a {
    pub type T = ();
    pub use crate::b::*;
}

use crate::a::T;

mod b {
    pub const T: () = ();
}
"#,
        expect![[r#"
            crate
            - T : type (import) value (import)
            - a : type
            - b : type

            crate::a
            - T : type value (glob)

            crate::b
            - T : value
        "#]],
    );
}

#[test]
fn underscore_import() {
    check(
        r#"
//- /main.rs
use tr::Tr as _;
use tr::Tr2 as _;

mod tr {
    pub trait Tr {}
    pub trait Tr2 {}
}
    "#,
        expect![[r#"
            crate
            - _ : type
            - _ : type
            - tr : type

            crate::tr
            - Tr : type
            - Tr2 : type
        "#]],
    );
}

#[test]
fn underscore_reexport() {
    check(
        r#"
//- /main.rs
mod tr {
    pub trait PubTr {}
    pub trait PrivTr {}
}
mod reex {
    use crate::tr::PrivTr as _;
    pub use crate::tr::PubTr as _;
}
use crate::reex::*;
    "#,
        expect![[r#"
            crate
            - _ : type
            - reex : type
            - tr : type

            crate::reex
            - _ : type
            - _ : type

            crate::tr
            - PrivTr : type
            - PubTr : type
        "#]],
    );
}

#[test]
fn underscore_pub_crate_reexport() {
    cov_mark::check!(upgrade_underscore_visibility);
    check(
        r#"
//- /main.rs crate:main deps:lib
use lib::*;

//- /lib.rs crate:lib
use tr::Tr as _;
pub use tr::Tr as _;

mod tr {
    pub trait Tr {}
}
    "#,
        expect![[r#"
            crate
            - _ : type
        "#]],
    );
}

#[test]
fn underscore_nontrait() {
    check(
        r#"
//- /main.rs
mod m {
    pub struct Struct;
    pub enum Enum {}
    pub const CONST: () = ();
}
use crate::m::{Struct as _, Enum as _, CONST as _};
    "#,
        expect![[r#"
            crate
            - m : type

            crate::m
            - CONST : value
            - Enum : type
            - Struct : type value
        "#]],
    );
}

#[test]
fn underscore_name_conflict() {
    check(
        r#"
//- /main.rs
struct Tr;

use tr::Tr as _;

mod tr {
    pub trait Tr {}
}
    "#,
        expect![[r#"
            crate
            - _ : type
            - Tr : type value
            - tr : type

            crate::tr
            - Tr : type
        "#]],
    );
}

#[test]
fn cfg_the_entire_crate() {
    check(
        r#"
//- /main.rs
#![cfg(never)]

pub struct S;
pub enum E {}
pub fn f() {}
    "#,
        expect![[r#"
            crate
        "#]],
    );
}

#[test]
fn use_crate_as() {
    check(
        r#"
use crate as foo;

use foo::bar as baz;

fn bar() {}
        "#,
        expect![[r#"
            crate
            - bar : value
            - baz : value (import)
            - foo : type (import)
        "#]],
    );
}

#[test]
fn self_imports_only_types() {
    check(
        r#"
//- /main.rs
mod m {
    pub macro S() {}
    pub struct S;
}

use self::m::S::{self};
    "#,
        expect![[r#"
            crate
            - S : type (import)
            - m : type

            crate::m
            - S : type value macro!
        "#]],
    );
}

#[test]
fn import_from_extern_crate_only_imports_public_items() {
    check(
        r#"
//- /lib.rs crate:lib deps:settings,macros
use macros::settings;
use settings::Settings;
//- /settings.rs crate:settings
pub struct Settings;
//- /macros.rs crate:macros
mod settings {}
pub const settings: () = ();
        "#,
        expect![[r#"
            crate
            - Settings : type (import) value (import)
            - settings : value (import)
        "#]],
    )
}

#[test]
fn non_prelude_deps() {
    check(
        r#"
//- /lib.rs crate:lib deps:dep extern-prelude:
use dep::Struct;
//- /dep.rs crate:dep
pub struct Struct;
        "#,
        expect![[r#"
            crate
            - Struct : _
        "#]],
    );
    check(
        r#"
//- /lib.rs crate:lib deps:dep extern-prelude:
extern crate dep;
use dep::Struct;
//- /dep.rs crate:dep
pub struct Struct;
        "#,
        expect![[r#"
            crate
            - Struct : type (import) value (import)
            - dep : type (extern)
        "#]],
    );
}

#[test]
fn braced_supers_in_use_tree() {
    cov_mark::check!(concat_super_mod_paths);
    check(
        r#"
mod some_module {
    pub fn unknown_func() {}
}

mod other_module {
    mod some_submodule {
        use { super::{ super::unknown_func, }, };
    }
}

use some_module::unknown_func;
        "#,
        expect![[r#"
            crate
            - other_module : type
            - some_module : type
            - unknown_func : value (import)

            crate::other_module
            - some_submodule : type

            crate::other_module::some_submodule
            - unknown_func : value (import)

            crate::some_module
            - unknown_func : value
        "#]],
    )
}
