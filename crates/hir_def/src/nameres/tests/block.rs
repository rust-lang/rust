use super::*;

#[test]
fn inner_item_smoke() {
    check_at(
        r#"
//- /lib.rs
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
//- /lib.rs
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
