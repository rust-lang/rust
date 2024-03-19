//! proc-macro tests

#[macro_use]
mod utils;
use utils::*;

use expect_test::expect;

#[test]
fn test_derive_empty() {
    assert_expand(
        "DeriveEmpty",
        r#"struct S;"#,
        expect!["SUBTREE $$ 1 1"],
        expect!["SUBTREE $$ 42:2@0..100#0 42:2@0..100#0"],
    );
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
                LITERAL "#[derive(DeriveError)] struct S ;"1
              PUNCH   ; [alone] 1"##]],
        expect![[r##"
            SUBTREE $$ 42:2@0..100#0 42:2@0..100#0
              IDENT   compile_error 42:2@0..100#0
              PUNCH   ! [alone] 42:2@0..100#0
              SUBTREE () 42:2@0..100#0 42:2@0..100#0
                LITERAL "#[derive(DeriveError)] struct S ;"42:2@0..100#0
              PUNCH   ; [alone] 42:2@0..100#0"##]],
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
              LITERAL 01
              PUNCH   , [alone] 1
              LITERAL 11
              PUNCH   , [alone] 1
              SUBTREE [] 1 1"#]],
        expect![[r#"
            SUBTREE $$ 42:2@0..100#0 42:2@0..100#0
              IDENT   ident 42:2@0..5#0
              PUNCH   , [alone] 42:2@5..6#0
              LITERAL 042:2@7..8#0
              PUNCH   , [alone] 42:2@8..9#0
              LITERAL 142:2@10..11#0
              PUNCH   , [alone] 42:2@11..12#0
              SUBTREE [] 42:2@13..14#0 42:2@14..15#0"#]],
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
        expect![[r#"
            SUBTREE $$ 42:2@0..100#0 42:2@0..100#0
              IDENT   ident 42:2@0..5#0
              PUNCH   , [alone] 42:2@5..6#0
              SUBTREE [] 42:2@7..8#0 42:2@7..8#0"#]],
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
        expect![[r#"
            SUBTREE $$ 42:2@0..100#0 42:2@0..100#0
              IDENT   r#async 42:2@0..7#0"#]],
    );
}

#[test]
fn test_fn_like_fn_like_span_join() {
    assert_expand(
        "fn_like_span_join",
        "foo     bar",
        expect![[r#"
            SUBTREE $$ 1 1
              IDENT   r#joined 1"#]],
        expect![[r#"
            SUBTREE $$ 42:2@0..100#0 42:2@0..100#0
              IDENT   r#joined 42:2@0..11#0"#]],
    );
}

#[test]
fn test_fn_like_fn_like_span_ops() {
    assert_expand(
        "fn_like_span_ops",
        "set_def_site resolved_at_def_site start_span",
        expect![[r#"
            SUBTREE $$ 1 1
              IDENT   set_def_site 0
              IDENT   resolved_at_def_site 1
              IDENT   start_span 1"#]],
        expect![[r#"
            SUBTREE $$ 42:2@0..100#0 42:2@0..100#0
              IDENT   set_def_site 41:1@0..150#0
              IDENT   resolved_at_def_site 42:2@13..33#0
              IDENT   start_span 42:2@34..34#0"#]],
    );
}

#[test]
fn test_fn_like_mk_literals() {
    assert_expand(
        "fn_like_mk_literals",
        r#""#,
        expect![[r#"
            SUBTREE $$ 1 1
              LITERAL b"byte_string"1
              LITERAL 'c'1
              LITERAL "string"1
              LITERAL 3.14f641
              LITERAL 3.141
              LITERAL 123i641
              LITERAL 1231"#]],
        expect![[r#"
            SUBTREE $$ 42:2@0..100#0 42:2@0..100#0
              LITERAL b"byte_string"42:2@0..100#0
              LITERAL 'c'42:2@0..100#0
              LITERAL "string"42:2@0..100#0
              LITERAL 3.14f6442:2@0..100#0
              LITERAL 3.1442:2@0..100#0
              LITERAL 123i6442:2@0..100#0
              LITERAL 12342:2@0..100#0"#]],
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
        expect![[r#"
            SUBTREE $$ 42:2@0..100#0 42:2@0..100#0
              IDENT   standard 42:2@0..100#0
              IDENT   r#raw 42:2@0..100#0"#]],
    );
}

#[test]
fn test_fn_like_macro_clone_literals() {
    assert_expand(
        "fn_like_clone_tokens",
        r###"1u16, 2_u32, -4i64, 3.14f32, "hello bridge", "suffixed"suffix, r##"raw"##, 'a', b'b', c"null""###,
        expect![[r###"
            SUBTREE $$ 1 1
              LITERAL 1u161
              PUNCH   , [alone] 1
              LITERAL 2_u321
              PUNCH   , [alone] 1
              PUNCH   - [alone] 1
              LITERAL 4i641
              PUNCH   , [alone] 1
              LITERAL 3.14f321
              PUNCH   , [alone] 1
              LITERAL "hello bridge"1
              PUNCH   , [alone] 1
              LITERAL "suffixed"suffix1
              PUNCH   , [alone] 1
              LITERAL r##"raw"##1
              PUNCH   , [alone] 1
              LITERAL 'a'1
              PUNCH   , [alone] 1
              LITERAL b'b'1
              PUNCH   , [alone] 1
              LITERAL c"null"1"###]],
        expect![[r###"
            SUBTREE $$ 42:2@0..100#0 42:2@0..100#0
              LITERAL 1u1642:2@0..4#0
              PUNCH   , [alone] 42:2@4..5#0
              LITERAL 2_u3242:2@6..11#0
              PUNCH   , [alone] 42:2@11..12#0
              PUNCH   - [alone] 42:2@13..14#0
              LITERAL 4i6442:2@14..18#0
              PUNCH   , [alone] 42:2@18..19#0
              LITERAL 3.14f3242:2@20..27#0
              PUNCH   , [alone] 42:2@27..28#0
              LITERAL "hello bridge"42:2@29..43#0
              PUNCH   , [alone] 42:2@43..44#0
              LITERAL "suffixed"suffix42:2@45..61#0
              PUNCH   , [alone] 42:2@61..62#0
              LITERAL r##"raw"##42:2@63..73#0
              PUNCH   , [alone] 42:2@73..74#0
              LITERAL 'a'42:2@75..78#0
              PUNCH   , [alone] 42:2@78..79#0
              LITERAL b'b'42:2@80..84#0
              PUNCH   , [alone] 42:2@84..85#0
              LITERAL c"null"42:2@86..93#0"###]],
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
                LITERAL "#[attr_error(some arguments)] mod m {}"1
              PUNCH   ; [alone] 1"##]],
        expect![[r##"
            SUBTREE $$ 42:2@0..100#0 42:2@0..100#0
              IDENT   compile_error 42:2@0..100#0
              PUNCH   ! [alone] 42:2@0..100#0
              SUBTREE () 42:2@0..100#0 42:2@0..100#0
                LITERAL "#[attr_error(some arguments)] mod m {}"42:2@0..100#0
              PUNCH   ; [alone] 42:2@0..100#0"##]],
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
        fn_like_span_join [FuncLike]
        fn_like_span_ops [FuncLike]
        attr_noop [Attr]
        attr_panic [Attr]
        attr_error [Attr]
        DeriveEmpty [CustomDerive]
        DerivePanic [CustomDerive]
        DeriveError [CustomDerive]"#]]
    .assert_eq(&res);
}
