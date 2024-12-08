use super::*;

#[test]
fn glob_1() {
    check(
        r#"
//- /lib.rs
mod foo;
use foo::*;

//- /foo/mod.rs
pub mod bar;
pub use self::bar::Baz;
pub struct Foo;

//- /foo/bar.rs
pub struct Baz;
"#,
        expect![[r#"
            crate
            Baz: t v
            Foo: t v
            bar: t
            foo: t

            crate::foo
            Baz: ti vi
            Foo: t v
            bar: t

            crate::foo::bar
            Baz: t v
        "#]],
    );
}

#[test]
fn glob_2() {
    check(
        r#"
//- /lib.rs
mod foo;
use foo::*;

//- /foo/mod.rs
pub mod bar;
pub use self::bar::*;
pub struct Foo;

//- /foo/bar.rs
pub struct Baz;
pub use super::*;
"#,
        expect![[r#"
            crate
            Baz: t v
            Foo: t v
            bar: t
            foo: t

            crate::foo
            Baz: t v
            Foo: t v
            bar: t

            crate::foo::bar
            Baz: t v
            Foo: t v
            bar: t
        "#]],
    );
}

#[test]
fn glob_privacy_1() {
    check(
        r"
//- /lib.rs
mod foo;
use foo::*;

//- /foo/mod.rs
pub mod bar;
pub use self::bar::*;
struct PrivateStructFoo;

//- /foo/bar.rs
pub struct Baz;
struct PrivateStructBar;
pub use super::*;
",
        expect![[r#"
            crate
            Baz: t v
            bar: t
            foo: t

            crate::foo
            Baz: t v
            PrivateStructFoo: t v
            bar: t

            crate::foo::bar
            Baz: t v
            PrivateStructBar: t v
            PrivateStructFoo: t v
            bar: t
        "#]],
    );
}

#[test]
fn glob_privacy_2() {
    check(
        r"
//- /lib.rs
mod foo;
use foo::*;
use foo::bar::*;

//- /foo/mod.rs
pub mod bar;
fn Foo() {};
pub struct Foo {};

//- /foo/bar.rs
pub(super) struct PrivateBaz;
struct PrivateBar;
pub(crate) struct PubCrateStruct;
",
        expect![[r#"
            crate
            Foo: t
            PubCrateStruct: t v
            bar: t
            foo: t

            crate::foo
            Foo: t v
            bar: t

            crate::foo::bar
            PrivateBar: t v
            PrivateBaz: t v
            PubCrateStruct: t v
        "#]],
    );
}

#[test]
fn glob_across_crates() {
    cov_mark::check!(glob_across_crates);
    check(
        r#"
//- /main.rs crate:main deps:test_crate
use test_crate::*;

//- /lib.rs crate:test_crate
pub struct Baz;
"#,
        expect![[r#"
            crate
            Baz: t v
        "#]],
    );
}

#[test]
fn glob_privacy_across_crates() {
    check(
        r#"
//- /main.rs crate:main deps:test_crate
use test_crate::*;

//- /lib.rs crate:test_crate
pub struct Baz;
struct Foo;
"#,
        expect![[r#"
            crate
            Baz: t v
        "#]],
    );
}

#[test]
fn glob_enum() {
    cov_mark::check!(glob_enum);
    check(
        r#"
enum Foo { Bar, Baz }
use self::Foo::*;
"#,
        expect![[r#"
            crate
            Bar: t v
            Baz: t v
            Foo: t
        "#]],
    );
}

#[test]
fn glob_enum_group() {
    cov_mark::check!(glob_enum_group);
    check(
        r#"
enum Foo { Bar, Baz }
use self::Foo::{*};
"#,
        expect![[r#"
            crate
            Bar: t v
            Baz: t v
            Foo: t
        "#]],
    );
}

#[test]
fn glob_shadowed_def() {
    cov_mark::check!(import_shadowed);
    check(
        r#"
//- /lib.rs
mod foo;
mod bar;
use foo::*;
use bar::baz;
use baz::Bar;

//- /foo.rs
pub mod baz { pub struct Foo; }

//- /bar.rs
pub mod baz { pub struct Bar; }
"#,
        expect![[r#"
            crate
            Bar: ti vi
            bar: t
            baz: ti
            foo: t

            crate::bar
            baz: t

            crate::bar::baz
            Bar: t v

            crate::foo
            baz: t

            crate::foo::baz
            Foo: t v
        "#]],
    );
}

#[test]
fn glob_shadowed_def_reversed() {
    check(
        r#"
//- /lib.rs
mod foo;
mod bar;
use bar::baz;
use foo::*;
use baz::Bar;

//- /foo.rs
pub mod baz { pub struct Foo; }

//- /bar.rs
pub mod baz { pub struct Bar; }
"#,
        expect![[r#"
            crate
            Bar: ti vi
            bar: t
            baz: ti
            foo: t

            crate::bar
            baz: t

            crate::bar::baz
            Bar: t v

            crate::foo
            baz: t

            crate::foo::baz
            Foo: t v
        "#]],
    );
}

#[test]
fn glob_shadowed_def_dependencies() {
    check(
        r#"
mod a { pub mod foo { pub struct X; } }
mod b { pub use super::a::foo; }
mod c { pub mod foo { pub struct Y; } }
mod d {
    use super::c::foo;
    use super::b::*;
    use foo::Y;
}
"#,
        expect![[r#"
            crate
            a: t
            b: t
            c: t
            d: t

            crate::a
            foo: t

            crate::a::foo
            X: t v

            crate::b
            foo: ti

            crate::c
            foo: t

            crate::c::foo
            Y: t v

            crate::d
            Y: ti vi
            foo: ti
        "#]],
    );
}

#[test]
fn glob_name_collision_check_visibility() {
    check(
        r#"
mod event {
    mod serenity {
        pub fn Event() {}
    }
    use serenity::*;

    pub struct Event {}
}

use event::Event;
        "#,
        expect![[r#"
            crate
            Event: ti
            event: t

            crate::event
            Event: t v
            serenity: t

            crate::event::serenity
            Event: v
        "#]],
    );
}

#[test]
fn glob_may_override_visibility() {
    check(
        r#"
mod reexport {
    use crate::defs::*;
    mod inner {
        pub use crate::defs::{Trait, function, makro};
    }
    pub use inner::*;
}
mod defs {
    pub trait Trait {}
    pub fn function() {}
    pub macro makro($t:item) { $t }
}
use reexport::*;
"#,
        expect![[r#"
            crate
            Trait: t
            defs: t
            function: v
            makro: m
            reexport: t

            crate::defs
            Trait: t
            function: v
            makro: m

            crate::reexport
            Trait: t
            function: v
            inner: t
            makro: m

            crate::reexport::inner
            Trait: ti
            function: vi
            makro: mi
        "#]],
    );
}

#[test]
fn regression_18308() {
    check(
        r#"
use outer::*;

mod outer {
    mod inner_superglob {
        pub use super::*;
    }

    // The importing order matters!
    pub use inner_superglob::*;
    use super::glob_target::*;
}

mod glob_target {
    pub struct ShouldBePrivate;
}
"#,
        expect![[r#"
            crate
            glob_target: t
            outer: t

            crate::glob_target
            ShouldBePrivate: t v

            crate::outer
            ShouldBePrivate: t v
            inner_superglob: t

            crate::outer::inner_superglob
            ShouldBePrivate: t v
            inner_superglob: t
        "#]],
    );
}
