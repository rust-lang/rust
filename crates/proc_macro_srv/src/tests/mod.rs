//! proc-macro tests

#[macro_use]
mod utils;
use expect_test::expect;
use paths::AbsPathBuf;
use utils::*;

#[test]
fn test_derive_empty() {
    assert_expand("DeriveEmpty", r#"struct S;"#, expect![[r#"SUBTREE $"#]]);
}

#[test]
fn test_derive_error() {
    assert_expand(
        "DeriveError",
        r#"struct S;"#,
        expect![[r##"
            SUBTREE $
              IDENT   compile_error 4294967295
              PUNCH   ! [alone] 4294967295
              SUBTREE () 4294967295
                LITERAL "#[derive(DeriveError)] struct S ;" 4294967295
              PUNCH   ; [alone] 4294967295"##]],
    );
}

#[test]
fn test_fn_like_macro() {
    assert_expand(
        "fn_like_noop",
        r#"ident, 0, 1, []"#,
        expect![[r#"
            SUBTREE $
              IDENT   ident 4294967295
              PUNCH   , [alone] 4294967295
              LITERAL 0 4294967295
              PUNCH   , [alone] 4294967295
              LITERAL 1 4294967295
              PUNCH   , [alone] 4294967295
              SUBTREE [] 4294967295"#]],
    );
}

#[test]
fn test_fn_like_macro2() {
    assert_expand(
        "fn_like_clone_tokens",
        r#"ident, []"#,
        expect![[r#"
            SUBTREE $
              IDENT   ident 4294967295
              PUNCH   , [alone] 4294967295
              SUBTREE [] 4294967295"#]],
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
            SUBTREE $
              IDENT   compile_error 4294967295
              PUNCH   ! [alone] 4294967295
              SUBTREE () 4294967295
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
        attr_noop [Attr]
        attr_panic [Attr]
        attr_error [Attr]
        DeriveEmpty [CustomDerive]
        DerivePanic [CustomDerive]
        DeriveError [CustomDerive]"#]]
    .assert_eq(&res);
}

#[test]
fn test_version_check() {
    let path = AbsPathBuf::assert(fixtures::proc_macro_test_dylib_path());
    let info = proc_macro_api::read_dylib_info(&path).unwrap();
    assert!(info.version.1 >= 50);
}
