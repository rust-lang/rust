//! proc-macro tests

#[macro_use]
mod utils;
use utils::*;

use expect_test::expect;

#[test]
fn test_derive_empty() {
    assert_expand("DeriveEmpty", r#"struct S;"#, expect!["SUBTREE $$ 1 1"]);
}

#[test]
fn test_derive_error() {
    assert_expand(
        "DeriveError",
        r#"struct S;"#,
        expect![[r##"
            SUBTREE $$ 1 1
              IDENT   compile_error 1
              PUNCH   ! [alone] 1
              SUBTREE () 1 1
                LITERAL "#[derive(DeriveError)] struct S ;" 1
              PUNCH   ; [alone] 1"##]],
    );
}

#[test]
fn test_fn_like_macro_noop() {
    assert_expand(
        "fn_like_noop",
        r#"ident, 0, 1, []"#,
        expect![[r#"
            SUBTREE $$ 1 1
              IDENT   ident 1
              PUNCH   , [alone] 1
              LITERAL 0 1
              PUNCH   , [alone] 1
              LITERAL 1 1
              PUNCH   , [alone] 1
              SUBTREE [] 1 1"#]],
    );
}

#[test]
fn test_fn_like_macro_clone_ident_subtree() {
    assert_expand(
        "fn_like_clone_tokens",
        r#"ident, []"#,
        expect![[r#"
            SUBTREE $$ 1 1
              IDENT   ident 1
              PUNCH   , [alone] 1
              SUBTREE [] 1 1"#]],
    );
}

#[test]
fn test_fn_like_macro_clone_raw_ident() {
    assert_expand(
        "fn_like_clone_tokens",
        "r#async",
        expect![[r#"
            SUBTREE $$ 1 1
              IDENT   r#async 1"#]],
    );
}

#[test]
fn test_fn_like_mk_literals() {
    assert_expand(
        "fn_like_mk_literals",
        r#""#,
        expect![[r#"
            SUBTREE $$ 1 1
              LITERAL b"byte_string" 1
              LITERAL 'c' 1
              LITERAL "string" 1
              LITERAL 3.14f64 1
              LITERAL 3.14 1
              LITERAL 123i64 1
              LITERAL 123 1"#]],
    );
}

#[test]
fn test_fn_like_mk_idents() {
    assert_expand(
        "fn_like_mk_idents",
        r#""#,
        expect![[r#"
            SUBTREE $$ 1 1
              IDENT   standard 1
              IDENT   r#raw 1"#]],
    );
}

#[test]
fn test_fn_like_macro_clone_literals() {
    assert_expand(
        "fn_like_clone_tokens",
        r#"1u16, 2_u32, -4i64, 3.14f32, "hello bridge""#,
        expect![[r#"
            SUBTREE $$ 1 1
              LITERAL 1u16 1
              PUNCH   , [alone] 1
              LITERAL 2_u32 1
              PUNCH   , [alone] 1
              PUNCH   - [alone] 1
              LITERAL 4i64 1
              PUNCH   , [alone] 1
              LITERAL 3.14f32 1
              PUNCH   , [alone] 1
              LITERAL "hello bridge" 1"#]],
    );
}

#[test]
fn test_attr_macro() {
    // Corresponds to
    //    #[proc_macro_test::attr_error(some arguments)]
    //    mod m {}
    assert_expand_attr(
        "attr_error",
        r#"mod m {}"#,
        r#"some arguments"#,
        expect![[r##"
            SUBTREE $$ 1 1
              IDENT   compile_error 1
              PUNCH   ! [alone] 1
              SUBTREE () 1 1
                LITERAL "#[attr_error(some arguments)] mod m {}" 1
              PUNCH   ; [alone] 1"##]],
    );
}

/// Tests that we find and classify all proc macros correctly.
#[test]
fn list_test_macros() {
    let res = list().join("\n");

    expect![[r#"
        fn_like_noop [FuncLike]
        fn_like_panic [FuncLike]
        fn_like_error [FuncLike]
        fn_like_clone_tokens [FuncLike]
        fn_like_mk_literals [FuncLike]
        fn_like_mk_idents [FuncLike]
        attr_noop [Attr]
        attr_panic [Attr]
        attr_error [Attr]
        DeriveEmpty [CustomDerive]
        DerivePanic [CustomDerive]
        DeriveError [CustomDerive]"#]]
    .assert_eq(&res);
}
