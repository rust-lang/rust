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
        expect!["SUBTREE $$ 42:2@0..100#2 42:2@0..100#2"],
    );
}

#[test]
fn test_derive_error() {
    assert_expand(
        "DeriveError",
        r#"struct S;"#,
        expect![[r#"
            SUBTREE $$ 1 1
              IDENT   compile_error 1
              PUNCH   ! [alone] 1
              SUBTREE () 1 1
                LITERAL Str #[derive(DeriveError)] struct S ; 1
              PUNCH   ; [alone] 1"#]],
        expect![[r#"
            SUBTREE $$ 42:2@0..100#2 42:2@0..100#2
              IDENT   compile_error 42:2@0..100#2
              PUNCH   ! [alone] 42:2@0..100#2
              SUBTREE () 42:2@0..100#2 42:2@0..100#2
                LITERAL Str #[derive(DeriveError)] struct S ; 42:2@0..100#2
              PUNCH   ; [alone] 42:2@0..100#2"#]],
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
              LITERAL Integer 0 1
              PUNCH   , [alone] 1
              LITERAL Integer 1 1
              PUNCH   , [alone] 1
              SUBTREE [] 1 1"#]],
        expect![[r#"
            SUBTREE $$ 42:2@0..100#2 42:2@0..100#2
              IDENT   ident 42:2@0..5#2
              PUNCH   , [alone] 42:2@5..6#2
              LITERAL Integer 0 42:2@7..8#2
              PUNCH   , [alone] 42:2@8..9#2
              LITERAL Integer 1 42:2@10..11#2
              PUNCH   , [alone] 42:2@11..12#2
              SUBTREE [] 42:2@13..14#2 42:2@14..15#2"#]],
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
            SUBTREE $$ 42:2@0..100#2 42:2@0..100#2
              IDENT   ident 42:2@0..5#2
              PUNCH   , [alone] 42:2@5..6#2
              SUBTREE [] 42:2@7..8#2 42:2@7..8#2"#]],
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
            SUBTREE $$ 42:2@0..100#2 42:2@0..100#2
              IDENT   r#async 42:2@0..7#2"#]],
    );
}

#[test]
#[cfg(not(bootstrap))]
fn test_fn_like_fn_like_span_join() {
    assert_expand(
        "fn_like_span_join",
        "foo     bar",
        expect![[r#"
            SUBTREE $$ 1 1
              IDENT   r#joined 1"#]],
        expect![[r#"
            SUBTREE $$ 42:2@0..100#2 42:2@0..100#2
              IDENT   r#joined 42:2@0..11#2"#]],
    );
}

#[test]
#[cfg(not(bootstrap))]
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
            SUBTREE $$ 42:2@0..100#2 42:2@0..100#2
              IDENT   set_def_site 41:1@0..150#2
              IDENT   resolved_at_def_site 42:2@13..33#2
              IDENT   start_span 42:2@34..34#2"#]],
    );
}

#[test]
fn test_fn_like_mk_literals() {
    assert_expand(
        "fn_like_mk_literals",
        r#""#,
        expect![[r#"
            SUBTREE $$ 1 1
              LITERAL ByteStr byte_string 1
              LITERAL Char c 1
              LITERAL Str string 1
              LITERAL Float 3.14f64 1
              LITERAL Float 3.14 1
              LITERAL Integer 123i64 1
              LITERAL Integer 123 1"#]],
        expect![[r#"
            SUBTREE $$ 42:2@0..100#2 42:2@0..100#2
              LITERAL ByteStr byte_string 42:2@0..100#2
              LITERAL Char c 42:2@0..100#2
              LITERAL Str string 42:2@0..100#2
              LITERAL Float 3.14f64 42:2@0..100#2
              LITERAL Float 3.14 42:2@0..100#2
              LITERAL Integer 123i64 42:2@0..100#2
              LITERAL Integer 123 42:2@0..100#2"#]],
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
            SUBTREE $$ 42:2@0..100#2 42:2@0..100#2
              IDENT   standard 42:2@0..100#2
              IDENT   r#raw 42:2@0..100#2"#]],
    );
}

#[test]
fn test_fn_like_macro_clone_literals() {
    assert_expand(
        "fn_like_clone_tokens",
        r###"1u16, 2_u32, -4i64, 3.14f32, "hello bridge", "suffixed"suffix, r##"raw"##, 'a', b'b', c"null""###,
        expect![[r#"
            SUBTREE $$ 1 1
              LITERAL Integer 1u16 1
              PUNCH   , [alone] 1
              LITERAL Integer 2_u32 1
              PUNCH   , [alone] 1
              PUNCH   - [alone] 1
              LITERAL Integer 4i64 1
              PUNCH   , [alone] 1
              LITERAL Float 3.14f32 1
              PUNCH   , [alone] 1
              LITERAL Str hello bridge 1
              PUNCH   , [alone] 1
              LITERAL Str suffixedsuffix 1
              PUNCH   , [alone] 1
              LITERAL StrRaw(2) raw 1
              PUNCH   , [alone] 1
              LITERAL Char a 1
              PUNCH   , [alone] 1
              LITERAL Byte b 1
              PUNCH   , [alone] 1
              LITERAL CStr null 1"#]],
        expect![[r#"
            SUBTREE $$ 42:2@0..100#2 42:2@0..100#2
              LITERAL Integer 1u16 42:2@0..4#2
              PUNCH   , [alone] 42:2@4..5#2
              LITERAL Integer 2_u32 42:2@6..11#2
              PUNCH   , [alone] 42:2@11..12#2
              PUNCH   - [alone] 42:2@13..14#2
              LITERAL Integer 4i64 42:2@14..18#2
              PUNCH   , [alone] 42:2@18..19#2
              LITERAL Float 3.14f32 42:2@20..27#2
              PUNCH   , [alone] 42:2@27..28#2
              LITERAL Str hello bridge 42:2@29..43#2
              PUNCH   , [alone] 42:2@43..44#2
              LITERAL Str suffixedsuffix 42:2@45..61#2
              PUNCH   , [alone] 42:2@61..62#2
              LITERAL StrRaw(2) raw 42:2@63..73#2
              PUNCH   , [alone] 42:2@73..74#2
              LITERAL Char a 42:2@75..78#2
              PUNCH   , [alone] 42:2@78..79#2
              LITERAL Byte b 42:2@80..84#2
              PUNCH   , [alone] 42:2@84..85#2
              LITERAL CStr null 42:2@86..93#2"#]],
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
        expect![[r#"
            SUBTREE $$ 1 1
              IDENT   compile_error 1
              PUNCH   ! [alone] 1
              SUBTREE () 1 1
                LITERAL Str #[attr_error(some arguments)] mod m {} 1
              PUNCH   ; [alone] 1"#]],
        expect![[r#"
            SUBTREE $$ 42:2@0..100#2 42:2@0..100#2
              IDENT   compile_error 42:2@0..100#2
              PUNCH   ! [alone] 42:2@0..100#2
              SUBTREE () 42:2@0..100#2 42:2@0..100#2
                LITERAL Str #[attr_error(some arguments)] mod m {} 42:2@0..100#2
              PUNCH   ; [alone] 42:2@0..100#2"#]],
    );
}

/// Tests that we find and classify all proc macros correctly.
#[test]
fn list_test_macros() {
    let res = list().join("\n");

    expect![[r#"
        fn_like_noop [Bang]
        fn_like_panic [Bang]
        fn_like_error [Bang]
        fn_like_clone_tokens [Bang]
        fn_like_mk_literals [Bang]
        fn_like_mk_idents [Bang]
        fn_like_span_join [Bang]
        fn_like_span_ops [Bang]
        attr_noop [Attr]
        attr_panic [Attr]
        attr_error [Attr]
        DeriveEmpty [CustomDerive]
        DerivePanic [CustomDerive]
        DeriveError [CustomDerive]"#]]
    .assert_eq(&res);
}
