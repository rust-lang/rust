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
            - Baz : type (glob) value (glob)
            - Foo : type (glob) value (glob)
            - bar : type (glob)
            - foo : type

            crate::foo
            - Baz : type (import) value (import)
            - Foo : type value
            - bar : type

            crate::foo::bar
            - Baz : type value
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
            - Baz : type (glob) value (glob)
            - Foo : type (glob) value (glob)
            - bar : type (glob)
            - foo : type

            crate::foo
            - Baz : type (glob) value (glob)
            - Foo : type value
            - bar : type

            crate::foo::bar
            - Baz : type value
            - Foo : type (glob) value (glob)
            - bar : type (glob)
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
            - Baz : type (glob) value (glob)
            - bar : type (glob)
            - foo : type

            crate::foo
            - Baz : type (glob) value (glob)
            - PrivateStructFoo : type value
            - bar : type

            crate::foo::bar
            - Baz : type value
            - PrivateStructBar : type value
            - PrivateStructFoo : type (glob) value (glob)
            - bar : type (glob)
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
            - Foo : type (glob)
            - PubCrateStruct : type (glob) value (glob)
            - bar : type (glob)
            - foo : type

            crate::foo
            - Foo : type value
            - bar : type

            crate::foo::bar
            - PrivateBar : type value
            - PrivateBaz : type value
            - PubCrateStruct : type value
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
            - Baz : type (glob) value (glob)
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
            - Baz : type (glob) value (glob)
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
            - Bar : type (glob) value (glob)
            - Baz : type (glob) value (glob)
            - Foo : type
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
            - Bar : type (glob) value (glob)
            - Baz : type (glob) value (glob)
            - Foo : type
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
            - Bar : type (import) value (import)
            - bar : type
            - baz : type (import)
            - foo : type

            crate::bar
            - baz : type

            crate::bar::baz
            - Bar : type value

            crate::foo
            - baz : type

            crate::foo::baz
            - Foo : type value
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
            - Bar : type (import) value (import)
            - bar : type
            - baz : type (import)
            - foo : type

            crate::bar
            - baz : type

            crate::bar::baz
            - Bar : type value

            crate::foo
            - baz : type

            crate::foo::baz
            - Foo : type value
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
            - a : type
            - b : type
            - c : type
            - d : type

            crate::a
            - foo : type

            crate::a::foo
            - X : type value

            crate::b
            - foo : type (import)

            crate::c
            - foo : type

            crate::c::foo
            - Y : type value

            crate::d
            - Y : type (import) value (import)
            - foo : type (import)
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
            - Event : type (import)
            - event : type

            crate::event
            - Event : type value (glob)
            - serenity : type

            crate::event::serenity
            - Event : value
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
            - Trait : type (glob)
            - defs : type
            - function : value (glob)
            - makro : macro! (glob)
            - reexport : type

            crate::defs
            - Trait : type
            - function : value
            - makro : macro!

            crate::reexport
            - Trait : type (glob)
            - function : value (glob)
            - inner : type
            - makro : macro! (glob)

            crate::reexport::inner
            - Trait : type (import)
            - function : value (import)
            - makro : macro! (import)
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
            - glob_target : type
            - outer : type

            crate::glob_target
            - ShouldBePrivate : type value

            crate::outer
            - ShouldBePrivate : type (glob) value (glob)
            - inner_superglob : type

            crate::outer::inner_superglob
            - ShouldBePrivate : type (glob) value (glob)
            - inner_superglob : type (glob)
        "#]],
    );
}

#[test]
fn regression_18580() {
    check(
        r#"
pub mod libs {
    pub struct Placeholder;
}

pub mod reexport_2 {
    use reexport_1::*;
    pub use reexport_1::*;

    pub mod reexport_1 {
        pub use crate::libs::*;
    }
}

use reexport_2::*;
"#,
        expect![[r#"
            crate
            - Placeholder : type (glob) value (glob)
            - libs : type
            - reexport_1 : type (glob)
            - reexport_2 : type

            crate::libs
            - Placeholder : type value

            crate::reexport_2
            - Placeholder : type (glob) value (glob)
            - reexport_1 : type

            crate::reexport_2::reexport_1
            - Placeholder : type (glob) value (glob)
        "#]],
    );
}
