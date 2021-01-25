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
