use super::*;

#[test]
fn name_res_works_for_broken_modules() {
    cov_mark::check!(name_res_works_for_broken_modules);
    check(
        r"
//- /lib.rs
mod foo // no `;`, no body
use self::foo::Baz;

//- /foo/mod.rs
pub mod bar;
pub use self::bar::Baz;

//- /foo/bar.rs
pub struct Baz;
",
        expect![[r#"
            crate
            - Baz : _
            - foo : type

            crate::foo
        "#]],
    );
}

#[test]
fn nested_module_resolution() {
    check(
        r#"
//- /lib.rs
mod n1;

//- /n1.rs
mod n2;

//- /n1/n2.rs
struct X;
"#,
        expect![[r#"
            crate
            - n1 : type

            crate::n1
            - n2 : type

            crate::n1::n2
            - X : type value
        "#]],
    );
}

#[test]
fn nested_module_resolution_2() {
    check(
        r#"
//- /lib.rs
mod prelude;
mod iter;

//- /prelude.rs
pub use crate::iter::Iterator;

//- /iter.rs
pub use self::traits::Iterator;
mod traits;

//- /iter/traits.rs
pub use self::iterator::Iterator;
mod iterator;

//- /iter/traits/iterator.rs
pub trait Iterator;
"#,
        expect![[r#"
            crate
            - iter : type
            - prelude : type

            crate::iter
            - Iterator : type (import)
            - traits : type

            crate::iter::traits
            - Iterator : type (import)
            - iterator : type

            crate::iter::traits::iterator
            - Iterator : type

            crate::prelude
            - Iterator : type (import)
        "#]],
    );
}

#[test]
fn module_resolution_works_for_non_standard_filenames() {
    check(
        r#"
//- /my_library.rs crate:my_library
mod foo;
use self::foo::Bar;

//- /foo/mod.rs
pub struct Bar;
"#,
        expect![[r#"
            crate
            - Bar : type (import) value (import)
            - foo : type

            crate::foo
            - Bar : type value
        "#]],
    );
}

#[test]
fn module_resolution_works_for_raw_modules() {
    check(
        r#"
//- /lib.rs
mod r#async;
use self::r#async::Bar;

//- /async.rs
mod foo;
mod r#async;
pub struct Bar;

//- /async/foo.rs
pub struct Foo;

//- /async/async.rs
pub struct Baz;
"#,
        expect![[r#"
            crate
            - Bar : type (import) value (import)
            - r#async : type

            crate::r#async
            - Bar : type value
            - r#async : type
            - foo : type

            crate::r#async::r#async
            - Baz : type value

            crate::r#async::foo
            - Foo : type value
        "#]],
    );
}

#[test]
fn module_resolution_works_for_inline_raw_modules() {
    check(
        r#"
//- /lib.rs
mod r#async {
    pub mod a;
    pub mod r#async;
}
use self::r#async::a::Foo;
use self::r#async::r#async::Bar;

//- /async/a.rs
pub struct Foo;

//- /async/async.rs
pub struct Bar;
"#,
        expect![[r#"
            crate
            - Bar : type (import) value (import)
            - Foo : type (import) value (import)
            - r#async : type

            crate::r#async
            - a : type
            - r#async : type

            crate::r#async::a
            - Foo : type value

            crate::r#async::r#async
            - Bar : type value
        "#]],
    );
}

#[test]
fn module_resolution_decl_path() {
    check(
        r#"
//- /lib.rs
#[path = "bar/baz/foo.rs"]
mod foo;
use self::foo::Bar;

//- /bar/baz/foo.rs
pub struct Bar;
"#,
        expect![[r#"
            crate
            - Bar : type (import) value (import)
            - foo : type

            crate::foo
            - Bar : type value
        "#]],
    );
}

#[test]
fn module_resolution_module_with_path_in_mod_rs() {
    check(
        r#"
//- /main.rs
mod foo;

//- /foo/mod.rs
#[path = "baz.rs"]
pub mod bar;
use self::bar::Baz;

//- /foo/baz.rs
pub struct Baz;
"#,
        expect![[r#"
            crate
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
fn module_resolution_module_with_path_non_crate_root() {
    check(
        r#"
//- /main.rs
mod foo;

//- /foo.rs
#[path = "baz.rs"]
pub mod bar;
use self::bar::Baz;

//- /baz.rs
pub struct Baz;
"#,
        expect![[r#"
            crate
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
fn module_resolution_module_decl_path_super() {
    check(
        r#"
//- /main.rs
#[path = "bar/baz/module.rs"]
mod foo;
pub struct Baz;

//- /bar/baz/module.rs
use super::Baz;
"#,
        expect![[r#"
            crate
            - Baz : type value
            - foo : type

            crate::foo
            - Baz : type (import) value (import)
        "#]],
    );
}

#[test]
fn module_resolution_explicit_path_mod_rs() {
    check(
        r#"
//- /main.rs
#[path = "module/mod.rs"]
mod foo;

//- /module/mod.rs
pub struct Baz;
"#,
        expect![[r#"
            crate
            - foo : type

            crate::foo
            - Baz : type value
        "#]],
    );
}

#[test]
fn module_resolution_relative_path() {
    check(
        r#"
//- /main.rs
mod foo;

//- /foo.rs
#[path = "./sub.rs"]
pub mod foo_bar;

//- /sub.rs
pub struct Baz;
"#,
        expect![[r#"
            crate
            - foo : type

            crate::foo
            - foo_bar : type

            crate::foo::foo_bar
            - Baz : type value
        "#]],
    );
}

#[test]
fn module_resolution_relative_path_2() {
    check(
        r#"
//- /main.rs
mod foo;

//- /foo/mod.rs
#[path="../sub.rs"]
pub mod foo_bar;

//- /sub.rs
pub struct Baz;
"#,
        expect![[r#"
            crate
            - foo : type

            crate::foo
            - foo_bar : type

            crate::foo::foo_bar
            - Baz : type value
        "#]],
    );
}

#[test]
fn module_resolution_relative_path_outside_root() {
    check(
        r#"
//- /a/b/c/d/e/main.rs crate:main
#[path="../../../../../outside.rs"]
mod foo;

//- /outside.rs
mod bar;

//- /bar.rs
pub struct Baz;
"#,
        expect![[r#"
            crate
            - foo : type

            crate::foo
            - bar : type

            crate::foo::bar
            - Baz : type value
        "#]],
    );
}

#[test]
fn module_resolution_explicit_path_mod_rs_2() {
    check(
        r#"
//- /main.rs
#[path = "module/bar/mod.rs"]
mod foo;

//- /module/bar/mod.rs
pub struct Baz;
"#,
        expect![[r#"
            crate
            - foo : type

            crate::foo
            - Baz : type value
        "#]],
    );
}

#[test]
fn module_resolution_explicit_path_mod_rs_with_win_separator() {
    check(
        r#"
//- /main.rs
#[path = r"module\bar\mod.rs"]
mod foo;

//- /module/bar/mod.rs
pub struct Baz;
"#,
        expect![[r#"
            crate
            - foo : type

            crate::foo
            - Baz : type value
        "#]],
    );
}

#[test]
fn module_resolution_decl_inside_inline_module_with_path_attribute() {
    check(
        r#"
//- /main.rs
#[path = "models"]
mod foo { mod bar; }

//- /models/bar.rs
pub struct Baz;
"#,
        expect![[r#"
            crate
            - foo : type

            crate::foo
            - bar : type

            crate::foo::bar
            - Baz : type value
        "#]],
    );
}

#[test]
fn module_resolution_decl_inside_inline_module() {
    check(
        r#"
//- /main.rs
mod foo { mod bar; }

//- /foo/bar.rs
pub struct Baz;
"#,
        expect![[r#"
            crate
            - foo : type

            crate::foo
            - bar : type

            crate::foo::bar
            - Baz : type value
        "#]],
    );
}

#[test]
fn module_resolution_decl_inside_inline_module_2_with_path_attribute() {
    check(
        r#"
//- /main.rs
#[path = "models/db"]
mod foo { mod bar; }

//- /models/db/bar.rs
pub struct Baz;
"#,
        expect![[r#"
            crate
            - foo : type

            crate::foo
            - bar : type

            crate::foo::bar
            - Baz : type value
        "#]],
    );
}

#[test]
fn module_resolution_decl_inside_inline_module_3() {
    check(
        r#"
//- /main.rs
#[path = "models/db"]
mod foo {
    #[path = "users.rs"]
    mod bar;
}

//- /models/db/users.rs
pub struct Baz;
"#,
        expect![[r#"
            crate
            - foo : type

            crate::foo
            - bar : type

            crate::foo::bar
            - Baz : type value
        "#]],
    );
}

#[test]
fn module_resolution_decl_inside_inline_module_empty_path() {
    check(
        r#"
//- /main.rs
#[path = ""]
mod foo {
    #[path = "users.rs"]
    mod bar;
}

//- /users.rs
pub struct Baz;
"#,
        expect![[r#"
            crate
            - foo : type

            crate::foo
            - bar : type

            crate::foo::bar
            - Baz : type value
        "#]],
    );
}

#[test]
fn module_resolution_decl_empty_path() {
    check(
        r#"
//- /main.rs
#[path = ""] // Should try to read `/` (a directory)
mod foo;

//- /foo.rs
pub struct Baz;
"#,
        expect![[r#"
            crate
            - foo : type

            crate::foo
        "#]],
    );
}

#[test]
fn module_resolution_decl_inside_inline_module_relative_path() {
    check(
        r#"
//- /main.rs
#[path = "./models"]
mod foo { mod bar; }

//- /models/bar.rs
pub struct Baz;
"#,
        expect![[r#"
            crate
            - foo : type

            crate::foo
            - bar : type

            crate::foo::bar
            - Baz : type value
        "#]],
    );
}

#[test]
fn module_resolution_decl_inside_inline_module_in_crate_root() {
    check(
        r#"
//- /main.rs
mod foo {
    #[path = "baz.rs"]
    pub mod bar;
}
use self::foo::bar::Baz;

//- /foo/baz.rs
pub struct Baz;
"#,
        expect![[r#"
            crate
            - Baz : type (import) value (import)
            - foo : type

            crate::foo
            - bar : type

            crate::foo::bar
            - Baz : type value
        "#]],
    );
}

#[test]
fn module_resolution_decl_inside_inline_module_in_mod_rs() {
    check(
        r#"
//- /main.rs
mod foo;

//- /foo/mod.rs
mod bar {
    #[path = "qwe.rs"]
    pub mod baz;
}
use self::bar::baz::Baz;

//- /foo/bar/qwe.rs
pub struct Baz;
"#,
        expect![[r#"
            crate
            - foo : type

            crate::foo
            - Baz : type (import) value (import)
            - bar : type

            crate::foo::bar
            - baz : type

            crate::foo::bar::baz
            - Baz : type value
        "#]],
    );
}

#[test]
fn module_resolution_decl_inside_inline_module_in_non_crate_root() {
    check(
        r#"
//- /main.rs
mod foo;

//- /foo.rs
mod bar {
    #[path = "qwe.rs"]
    pub mod baz;
}
use self::bar::baz::Baz;

//- /foo/bar/qwe.rs
pub struct Baz;
"#,
        expect![[r#"
            crate
            - foo : type

            crate::foo
            - Baz : type (import) value (import)
            - bar : type

            crate::foo::bar
            - baz : type

            crate::foo::bar::baz
            - Baz : type value
        "#]],
    );
}

#[test]
fn module_resolution_decl_inside_inline_module_in_non_crate_root_2() {
    check(
        r#"
//- /main.rs
mod foo;

//- /foo.rs
#[path = "bar"]
mod bar {
    pub mod baz;
}
use self::bar::baz::Baz;

//- /bar/baz.rs
pub struct Baz;
"#,
        expect![[r#"
            crate
            - foo : type

            crate::foo
            - Baz : type (import) value (import)
            - bar : type

            crate::foo::bar
            - baz : type

            crate::foo::bar::baz
            - Baz : type value
        "#]],
    );
}

#[test]
fn module_resolution_decl_inside_module_in_non_crate_root_2() {
    check(
        r#"
//- /main.rs
#[path="module/m2.rs"]
mod module;

//- /module/m2.rs
pub mod submod;

//- /module/submod.rs
pub struct Baz;
"#,
        expect![[r#"
            crate
            - module : type

            crate::module
            - submod : type

            crate::module::submod
            - Baz : type value
        "#]],
    );
}

#[test]
fn nested_out_of_line_module() {
    check(
        r#"
//- /lib.rs
mod a {
    mod b {
        mod c;
    }
}

//- /a/b/c.rs
struct X;
"#,
        expect![[r#"
            crate
            - a : type

            crate::a
            - b : type

            crate::a::b
            - c : type

            crate::a::b::c
            - X : type value
        "#]],
    );
}

#[test]
fn nested_out_of_line_module_with_path() {
    check(
        r#"
//- /lib.rs
mod a {
    #[path = "d/e"]
    mod b {
        mod c;
    }
}

//- /a/d/e/c.rs
struct X;
"#,
        expect![[r#"
            crate
            - a : type

            crate::a
            - b : type

            crate::a::b
            - c : type

            crate::a::b::c
            - X : type value
        "#]],
    );
}

#[test]
fn circular_mods() {
    cov_mark::check!(circular_mods);
    compute_crate_def_map(
        r#"
//- /lib.rs
mod foo;
//- /foo.rs
#[path = "./foo.rs"]
mod foo;
"#,
        |_| (),
    );

    compute_crate_def_map(
        r#"
//- /lib.rs
mod foo;
//- /foo.rs
#[path = "./bar.rs"]
mod bar;
//- /bar.rs
#[path = "./foo.rs"]
mod foo;
"#,
        |_| (),
    );
}

#[test]
fn abs_path_ignores_local() {
    check(
        r#"
//- /main.rs crate:main deps:core
pub use ::core::hash::Hash;
pub mod core {}

//- /lib.rs crate:core
pub mod hash { pub trait Hash {} }
"#,
        expect![[r#"
            crate
            - Hash : type (import)
            - core : type

            crate::core
        "#]],
    );
}

#[test]
fn cfg_in_module_file() {
    // Inner `#![cfg]` in a module file makes the whole module disappear.
    check(
        r#"
//- /main.rs
mod module;

//- /module.rs
#![cfg(NEVER)]

struct AlsoShouldNotAppear;
        "#,
        expect![[r#"
            crate
        "#]],
    )
}

#[test]
fn invalid_imports() {
    check(
        r#"
//- /main.rs
mod module;

use self::module::S::new;
use self::module::unresolved;
use self::module::C::const_based;
use self::module::Enum::Variant::NoAssoc;

//- /module.rs
pub struct S;
impl S {
    pub fn new() {}
}
pub const C: () = ();
pub enum Enum {
    Variant,
}
        "#,
        expect![[r#"
            crate
            - NoAssoc : _
            - const_based : _
            - module : type
            - new : _
            - unresolved : _

            crate::module
            - C : value
            - Enum : type
            - S : type value
        "#]],
    );
}

#[test]
fn trait_item_imports_same_crate() {
    check(
        r#"
//- /main.rs
mod module;

use self::module::Trait::{AssocType, ASSOC_CONST, MACRO_CONST, method};

//- /module.rs
macro_rules! m {
    ($name:ident) => { const $name: () = (); };
}
pub trait Trait {
    type AssocType;
    const ASSOC_CONST: ();
    fn method(&self);
    m!(MACRO_CONST);
}
        "#,
        expect![[r#"
            crate
            - ASSOC_CONST : _
            - AssocType : _
            - MACRO_CONST : _
            - method : _
            - module : type

            crate::module
            - Trait : type
            - (legacy) m : macro!
        "#]],
    );
    check(
        r#"
//- /main.rs
mod module;

use self::module::Trait::*;

//- /module.rs
macro_rules! m {
    ($name:ident) => { const $name: () = (); };
}
pub trait Trait {
    type AssocType;
    const ASSOC_CONST: ();
    fn method(&self);
    m!(MACRO_CONST);
}
        "#,
        expect![[r#"
            crate
            - module : type

            crate::module
            - Trait : type
            - (legacy) m : macro!
        "#]],
    );
}

#[test]
fn trait_item_imports_differing_crate() {
    check(
        r#"
//- /main.rs deps:lib crate:main
use lib::Trait::{AssocType, ASSOC_CONST, MACRO_CONST, method};

//- /lib.rs crate:lib
macro_rules! m {
    ($name:ident) => { const $name: () = (); };
}
pub trait Trait {
    type AssocType;
    const ASSOC_CONST: ();
    fn method(&self);
    m!(MACRO_CONST);
}
        "#,
        expect![[r#"
            crate
            - ASSOC_CONST : _
            - AssocType : _
            - MACRO_CONST : _
            - method : _
        "#]],
    );
    check(
        r#"
//- /main.rs deps:lib crate:main
use lib::Trait::*;

//- /lib.rs crate:lib
macro_rules! m {
    ($name:ident) => { const $name: () = (); };
}
pub trait Trait {
    type AssocType;
    const ASSOC_CONST: ();
    fn method(&self);
    m!(MACRO_CONST);
}
        "#,
        expect![[r#"
            crate
        "#]],
    );
}
