//! proc-macro tests

#[macro_use]
mod utils;
use utils::*;

use expect_test::expect;

#[test]
fn test_derive_empty() {
    assert_expand("DeriveEmpty", r#"struct S;"#, expect!["SUBTREE $$ 4294967295 4294967295"]);
}

#[test]
fn test_derive_error() {
    assert_expand(
        "DeriveError",
        r#"struct S;"#,
        expect![[r##"
            SUBTREE $$ 4294967295 4294967295
              IDENT   compile_error 4294967295
              PUNCH   ! [alone] 4294967295
              SUBTREE () 4294967295 4294967295
                LITERAL "#[derive(DeriveError)] struct S ;" 4294967295
              PUNCH   ; [alone] 4294967295"##]],
    );
}

#[test]
fn test_fn_like_macro_noop() {
    assert_expand(
        "fn_like_noop",
        r#"ident, 0, 1, []"#,
        expect![[r#"
            SUBTREE $$ 4294967295 4294967295
              IDENT   ident 4294967295
              PUNCH   , [alone] 4294967295
              LITERAL 0 4294967295
              PUNCH   , [alone] 4294967295
              LITERAL 1 4294967295
              PUNCH   , [alone] 4294967295
              SUBTREE [] 4294967295 4294967295"#]],
    );
}

#[test]
fn test_fn_like_macro_clone_ident_subtree() {
    assert_expand(
        "fn_like_clone_tokens",
        r#"ident, []"#,
        expect![[r#"
            SUBTREE $$ 4294967295 4294967295
              IDENT   ident 4294967295
              PUNCH   , [alone] 4294967295
              SUBTREE [] 4294967295 4294967295"#]],
    );
}

#[test]
fn test_fn_like_macro_clone_raw_ident() {
    assert_expand(
        "fn_like_clone_tokens",
        "r#async",
        expect![[r#"
            SUBTREE $$ 4294967295 4294967295
              IDENT   r#async 4294967295"#]],
    );
}

#[test]
fn test_fn_like_mk_literals() {
    assert_expand(
        "fn_like_mk_literals",
        r#""#,
        expect![[r#"
            SUBTREE $$ 4294967295 4294967295
              LITERAL b"byte_string" 4294967295
              LITERAL 'c' 4294967295
              LITERAL "string" 4294967295
              LITERAL 3.14f64 4294967295
              LITERAL 3.14 4294967295
              LITERAL 123i64 4294967295
              LITERAL 123 4294967295"#]],
    );
}

#[test]
fn test_fn_like_mk_idents() {
    assert_expand(
        "fn_like_mk_idents",
        r#""#,
        expect![[r#"
            SUBTREE $$ 4294967295 4294967295
              IDENT   standard 4294967295
              IDENT   r#raw 4294967295"#]],
    );
}

#[test]
fn test_fn_like_macro_clone_literals() {
    assert_expand(
        "fn_like_clone_tokens",
        r#"1u16, 2_u32, -4i64, 3.14f32, "hello bridge""#,
        expect![[r#"
            SUBTREE $$ 4294967295 4294967295
              LITERAL 1u16 4294967295
              PUNCH   , [alone] 4294967295
              LITERAL 2_u32 4294967295
              PUNCH   , [alone] 4294967295
              PUNCH   - [alone] 4294967295
              LITERAL 4i64 4294967295
              PUNCH   , [alone] 4294967295
              LITERAL 3.14f32 4294967295
              PUNCH   , [alone] 4294967295
              LITERAL "hello bridge" 4294967295"#]],
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
            SUBTREE $$ 4294967295 4294967295
              IDENT   compile_error 4294967295
              PUNCH   ! [alone] 4294967295
              SUBTREE () 4294967295 4294967295
                LITERAL "#[attr_error(some arguments)] mod m {}" 4294967295
              PUNCH   ; [alone] 4294967295"##]],
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
