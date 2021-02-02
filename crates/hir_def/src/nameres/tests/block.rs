use super::*;

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
struct Struct;
fn outer() {
    use Struct;
    use crate::Struct as CrateStruct;
    use self::Struct as SelfStruct;
    $0
}
"#,
        expect![[r#"
            block scope
            CrateStruct: t v
            SelfStruct: t v
            Struct: t v
            crate
            Struct: t v
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
use core::mark;

fn f() {
    fn nested() {
        mark::hit!(Hit);
        $0
    }
}
//- /core.rs crate:core
pub mod mark {
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
            f: v
            mark: t
        "#]],
    );
}
