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
fn nested_module_scoping() {
    check_block_scopes_at(
        r#"
fn f() {
    mod module {
        struct Struct {}
        fn f() {
            use self::Struct;
            $0
        }
    }
}
    "#,
        expect![[r#"
            BlockId(1) in ModuleId { krate: CrateId(0), block: Some(BlockId(0)), local_id: Idx::<ModuleData>(1) }
            BlockId(0) in ModuleId { krate: CrateId(0), block: None, local_id: Idx::<ModuleData>(0) }
            crate scope
        "#]],
    );
}

#[test]
fn legacy_macro_items() {
    // Checks that legacy-scoped `macro_rules!` from parent namespaces are resolved and expanded
    // correctly.
    check_at(
        r#"
macro_rules! mark {
    () => {
        struct Hit {}
    }
}

fn f() {
    mark!();
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
        cov_mark::mark!(Hit);
        $0
    }
}
//- /core.rs crate:core
pub mod cov_mark {
    #[macro_export]
    macro_rules! _mark {
        ($name:ident) => {
            struct $name {}
        }
    }

    pub use crate::_mark as mark;
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

#[test]
fn nested_macro_item_decl() {
    cov_mark::check!(macro_call_in_macro_stmts_is_added_to_item_tree);
    check_at(
        r#"
macro_rules! inner_declare {
    ($ident:ident) => {
        static $ident: u32 = 0;
    };
}
macro_rules! declare {
    ($ident:ident) => {
        inner_declare!($ident);
    };
}

fn foo() {
    declare!(bar);
    bar;
    $0
}
        "#,
        expect![[r#"
            block scope
            bar: v

            crate
            foo: v
        "#]],
    )
}

#[test]
fn is_visible_from_same_def_map() {
    // Regression test for https://github.com/rust-lang/rust-analyzer/issues/9481
    cov_mark::check!(is_visible_from_same_block_def_map);
    check_at(
        r#"
fn outer() {
    mod tests {
        use super::*;
    }
    use crate::name;
    $0
}
        "#,
        expect![[r#"
            block scope
            name: _
            tests: t

            block scope::tests
            name: _
            outer: v

            crate
            outer: v
        "#]],
    );
}

#[test]
fn stmt_macro_expansion_with_trailing_expr() {
    cov_mark::check!(macro_stmt_with_trailing_macro_expr);
    check_at(
        r#"
macro_rules! mac {
    () => { mac!($) };
    ($x:tt) => { fn inner() {} };
}
fn foo() {
    mac!();
    $0
}
        "#,
        expect![[r#"
            block scope
            inner: v

            crate
            foo: v
        "#]],
    )
}

#[test]
fn trailing_expr_macro_expands_stmts() {
    check_at(
        r#"
macro_rules! foo {
    () => { const FOO: u32 = 0;const BAR: u32 = 0; };
}
fn f() {$0
    foo!{}
};
        "#,
        expect![[r#"
            block scope
            BAR: v
            FOO: v

            crate
            f: v
        "#]],
    )
}
