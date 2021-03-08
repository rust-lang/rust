use super::*;
use expect_test::expect;

#[test]
fn inner_item_smoke() {
    check_at(
        r#"
struct inner {}
fn outer() {
    $0
    fn inner() {}
}
"#,
        expect![[r#"
            block scope
            inner: v

            crate
            inner: t
            outer: v
        "#]],
    );
}

#[test]
fn use_from_crate() {
    check_at(
        r#"
struct Struct {}
fn outer() {
    fn Struct() {}
    use Struct as PlainStruct;
    use crate::Struct as CrateStruct;
    use self::Struct as SelfStruct;
    use super::Struct as SuperStruct;
    $0
}
"#,
        expect![[r#"
            block scope
            CrateStruct: t
            PlainStruct: t v
            SelfStruct: t
            Struct: v
            SuperStruct: _

            crate
            Struct: t
            outer: v
        "#]],
    );
}

#[test]
fn merge_namespaces() {
    check_at(
        r#"
struct name {}
fn outer() {
    fn name() {}

    use name as imported; // should import both `name`s

    $0
}
"#,
        expect![[r#"
            block scope
            imported: t v
            name: v

            crate
            name: t
            outer: v
        "#]],
    );
}

#[test]
fn nested_blocks() {
    check_at(
        r#"
fn outer() {
    struct inner1 {}
    fn inner() {
        use inner1;
        use outer;
        fn inner2() {}
        $0
    }
}
"#,
        expect![[r#"
            block scope
            inner1: t
            inner2: v
            outer: v

            block scope
            inner: v
            inner1: t

            crate
            outer: v
        "#]],
    );
}

#[test]
fn super_imports() {
    check_at(
        r#"
mod module {
    fn f() {
        use super::Struct;
        $0
    }
}

struct Struct {}
"#,
        expect![[r#"
            block scope
            Struct: t

            crate
            Struct: t
            module: t

            crate::module
            f: v
        "#]],
    );
}

#[test]
fn legacy_macro_items() {
    // Checks that legacy-scoped `macro_rules!` from parent namespaces are resolved and expanded
    // correctly.
    check_at(
        r#"
macro_rules! hit {
    () => {
        struct Hit {}
    }
}

fn f() {
    hit!();
    $0
}
"#,
        expect![[r#"
            block scope
            Hit: t

            crate
            f: v
        "#]],
    );
}

#[test]
fn macro_resolve() {
    check_at(
        r#"
//- /lib.rs crate:lib deps:core
use core::cov_mark;

fn f() {
    fn nested() {
        cov_mark::hit!(Hit);
        $0
    }
}
//- /core.rs crate:core
pub mod cov_mark {
    #[macro_export]
    macro_rules! _hit {
        ($name:ident) => {
            struct $name {}
        }
    }

    pub use crate::_hit as hit;
}
"#,
        expect![[r#"
            block scope
            Hit: t

            block scope
            nested: v

            crate
            cov_mark: t
            f: v
        "#]],
    );
}

#[test]
fn macro_resolve_legacy() {
    check_at(
        r#"
//- /lib.rs
mod module;

//- /module.rs
macro_rules! m {
    () => {
        struct Def {}
    };
}

fn f() {
    {
        m!();
        $0
    }
}
        "#,
        expect![[r#"
            block scope
            Def: t

            crate
            module: t

            crate::module
            f: v
        "#]],
    )
}

#[test]
fn super_does_not_resolve_to_block_module() {
    check_at(
        r#"
fn main() {
    struct Struct {}
    mod module {
        use super::Struct;

        $0
    }
}
    "#,
        expect![[r#"
        block scope
        Struct: t
        module: t

        block scope::module
        Struct: _

        crate
        main: v
    "#]],
    );
}

#[test]
fn underscore_import() {
    // This used to panic, because the default (private) visibility inside block expressions would
    // point into the containing `DefMap`, which visibilities should never be able to do.
    cov_mark::check!(adjust_vis_in_block_def_map);
    check_at(
        r#"
mod m {
    fn main() {
        use Tr as _;
        trait Tr {}
        $0
    }
}
    "#,
        expect![[r#"
        block scope
        _: t
        Tr: t

        crate
        m: t

        crate::m
        main: v
    "#]],
    );
}
