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
        expect![[r#"
            SUBTREE $$ 1 1
              IDENT   struct 1
              IDENT   S 1
              PUNCH   ; [alone] 1



            SUBTREE $$ 1 1"#]],
        expect![[r#"
            SUBTREE $$ 42:Root[0000, 0]@0..100#ROOT2024 42:Root[0000, 0]@0..100#ROOT2024
              IDENT   struct 42:Root[0000, 0]@0..6#ROOT2024
              IDENT   S 42:Root[0000, 0]@7..8#ROOT2024
              PUNCH   ; [alone] 42:Root[0000, 0]@8..9#ROOT2024



            SUBTREE $$ 42:Root[0000, 0]@0..100#ROOT2024 42:Root[0000, 0]@0..100#ROOT2024"#]],
    );
}

#[test]
fn test_derive_error() {
    assert_expand(
        "DeriveError",
        r#"struct S;"#,
        expect![[r#"
            SUBTREE $$ 1 1
              IDENT   struct 1
              IDENT   S 1
              PUNCH   ; [alone] 1



            SUBTREE $$ 1 1
              IDENT   compile_error 1
              PUNCH   ! [alone] 1
              SUBTREE () 1 1
                LITERAL Str #[derive(DeriveError)] struct S ; 1
              PUNCH   ; [alone] 1"#]],
        expect![[r#"
            SUBTREE $$ 42:Root[0000, 0]@0..100#ROOT2024 42:Root[0000, 0]@0..100#ROOT2024
              IDENT   struct 42:Root[0000, 0]@0..6#ROOT2024
              IDENT   S 42:Root[0000, 0]@7..8#ROOT2024
              PUNCH   ; [alone] 42:Root[0000, 0]@8..9#ROOT2024



            SUBTREE $$ 42:Root[0000, 0]@0..100#ROOT2024 42:Root[0000, 0]@0..100#ROOT2024
              IDENT   compile_error 42:Root[0000, 0]@0..100#ROOT2024
              PUNCH   ! [alone] 42:Root[0000, 0]@0..100#ROOT2024
              SUBTREE () 42:Root[0000, 0]@0..100#ROOT2024 42:Root[0000, 0]@0..100#ROOT2024
                LITERAL Str #[derive(DeriveError)] struct S ; 42:Root[0000, 0]@0..100#ROOT2024
              PUNCH   ; [alone] 42:Root[0000, 0]@0..100#ROOT2024"#]],
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
              SUBTREE [] 1 1



            SUBTREE $$ 1 1
              IDENT   ident 1
              PUNCH   , [alone] 1
              LITERAL Integer 0 1
              PUNCH   , [alone] 1
              LITERAL Integer 1 1
              PUNCH   , [alone] 1
              SUBTREE [] 1 1"#]],
        expect![[r#"
            SUBTREE $$ 42:Root[0000, 0]@0..100#ROOT2024 42:Root[0000, 0]@0..100#ROOT2024
              IDENT   ident 42:Root[0000, 0]@0..5#ROOT2024
              PUNCH   , [alone] 42:Root[0000, 0]@5..6#ROOT2024
              LITERAL Integer 0 42:Root[0000, 0]@7..8#ROOT2024
              PUNCH   , [alone] 42:Root[0000, 0]@8..9#ROOT2024
              LITERAL Integer 1 42:Root[0000, 0]@10..11#ROOT2024
              PUNCH   , [alone] 42:Root[0000, 0]@11..12#ROOT2024
              SUBTREE [] 42:Root[0000, 0]@13..14#ROOT2024 42:Root[0000, 0]@14..15#ROOT2024



            SUBTREE $$ 42:Root[0000, 0]@0..100#ROOT2024 42:Root[0000, 0]@0..100#ROOT2024
              IDENT   ident 42:Root[0000, 0]@0..5#ROOT2024
              PUNCH   , [alone] 42:Root[0000, 0]@5..6#ROOT2024
              LITERAL Integer 0 42:Root[0000, 0]@7..8#ROOT2024
              PUNCH   , [alone] 42:Root[0000, 0]@8..9#ROOT2024
              LITERAL Integer 1 42:Root[0000, 0]@10..11#ROOT2024
              PUNCH   , [alone] 42:Root[0000, 0]@11..12#ROOT2024
              SUBTREE [] 42:Root[0000, 0]@13..14#ROOT2024 42:Root[0000, 0]@14..15#ROOT2024"#]],
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
              SUBTREE [] 1 1



            SUBTREE $$ 1 1
              IDENT   ident 1
              PUNCH   , [alone] 1
              SUBTREE [] 1 1"#]],
        expect![[r#"
            SUBTREE $$ 42:Root[0000, 0]@0..100#ROOT2024 42:Root[0000, 0]@0..100#ROOT2024
              IDENT   ident 42:Root[0000, 0]@0..5#ROOT2024
              PUNCH   , [alone] 42:Root[0000, 0]@5..6#ROOT2024
              SUBTREE [] 42:Root[0000, 0]@7..8#ROOT2024 42:Root[0000, 0]@8..9#ROOT2024



            SUBTREE $$ 42:Root[0000, 0]@0..100#ROOT2024 42:Root[0000, 0]@0..100#ROOT2024
              IDENT   ident 42:Root[0000, 0]@0..5#ROOT2024
              PUNCH   , [alone] 42:Root[0000, 0]@5..6#ROOT2024
              SUBTREE [] 42:Root[0000, 0]@7..9#ROOT2024 42:Root[0000, 0]@7..9#ROOT2024"#]],
    );
}

#[test]
fn test_fn_like_macro_clone_raw_ident() {
    assert_expand(
        "fn_like_clone_tokens",
        "r#async",
        expect![[r#"
            SUBTREE $$ 1 1
              IDENT   r#async 1



            SUBTREE $$ 1 1
              IDENT   r#async 1"#]],
        expect![[r#"
            SUBTREE $$ 42:Root[0000, 0]@0..100#ROOT2024 42:Root[0000, 0]@0..100#ROOT2024
              IDENT   r#async 42:Root[0000, 0]@0..7#ROOT2024



            SUBTREE $$ 42:Root[0000, 0]@0..100#ROOT2024 42:Root[0000, 0]@0..100#ROOT2024
              IDENT   r#async 42:Root[0000, 0]@0..7#ROOT2024"#]],
    );
}

#[test]
fn test_fn_like_fn_like_span_join() {
    assert_expand(
        "fn_like_span_join",
        "foo     bar",
        expect![[r#"
            SUBTREE $$ 1 1
              IDENT   foo 1
              IDENT   bar 1



            SUBTREE $$ 1 1
              IDENT   r#joined 1"#]],
        expect![[r#"
            SUBTREE $$ 42:Root[0000, 0]@0..100#ROOT2024 42:Root[0000, 0]@0..100#ROOT2024
              IDENT   foo 42:Root[0000, 0]@0..3#ROOT2024
              IDENT   bar 42:Root[0000, 0]@8..11#ROOT2024



            SUBTREE $$ 42:Root[0000, 0]@0..100#ROOT2024 42:Root[0000, 0]@0..100#ROOT2024
              IDENT   r#joined 42:Root[0000, 0]@0..11#ROOT2024"#]],
    );
}

#[test]
fn test_fn_like_fn_like_span_ops() {
    assert_expand(
        "fn_like_span_ops",
        "set_def_site resolved_at_def_site start_span",
        expect![[r#"
            SUBTREE $$ 1 1
              IDENT   set_def_site 1
              IDENT   resolved_at_def_site 1
              IDENT   start_span 1



            SUBTREE $$ 1 1
              IDENT   set_def_site 0
              IDENT   resolved_at_def_site 1
              IDENT   start_span 1"#]],
        expect![[r#"
            SUBTREE $$ 42:Root[0000, 0]@0..100#ROOT2024 42:Root[0000, 0]@0..100#ROOT2024
              IDENT   set_def_site 42:Root[0000, 0]@0..12#ROOT2024
              IDENT   resolved_at_def_site 42:Root[0000, 0]@13..33#ROOT2024
              IDENT   start_span 42:Root[0000, 0]@34..44#ROOT2024



            SUBTREE $$ 42:Root[0000, 0]@0..100#ROOT2024 42:Root[0000, 0]@0..100#ROOT2024
              IDENT   set_def_site 41:Root[0000, 0]@0..150#ROOT2024
              IDENT   resolved_at_def_site 42:Root[0000, 0]@13..33#ROOT2024
              IDENT   start_span 42:Root[0000, 0]@34..34#ROOT2024"#]],
    );
}

#[test]
fn test_fn_like_mk_literals() {
    assert_expand(
        "fn_like_mk_literals",
        r#""#,
        expect![[r#"
            SUBTREE $$ 1 1



            SUBTREE $$ 1 1
              LITERAL ByteStr byte_string 1
              LITERAL Char c 1
              LITERAL Str string 1
              LITERAL Str -string 1
              LITERAL CStr cstring 1
              LITERAL Float 3.14f64 1
              PUNCH   - [alone] 1
              LITERAL Float 3.14f64 1
              LITERAL Float 3.14 1
              PUNCH   - [alone] 1
              LITERAL Float 3.14 1
              LITERAL Integer 123i64 1
              PUNCH   - [alone] 1
              LITERAL Integer 123i64 1
              LITERAL Integer 123 1
              PUNCH   - [alone] 1
              LITERAL Integer 123 1"#]],
        expect![[r#"
            SUBTREE $$ 42:Root[0000, 0]@0..100#ROOT2024 42:Root[0000, 0]@0..100#ROOT2024



            SUBTREE $$ 42:Root[0000, 0]@0..100#ROOT2024 42:Root[0000, 0]@0..100#ROOT2024
              LITERAL ByteStr byte_string 42:Root[0000, 0]@0..100#ROOT2024
              LITERAL Char c 42:Root[0000, 0]@0..100#ROOT2024
              LITERAL Str string 42:Root[0000, 0]@0..100#ROOT2024
              LITERAL Str -string 42:Root[0000, 0]@0..100#ROOT2024
              LITERAL CStr cstring 42:Root[0000, 0]@0..100#ROOT2024
              LITERAL Float 3.14f64 42:Root[0000, 0]@0..100#ROOT2024
              PUNCH   - [alone] 42:Root[0000, 0]@0..100#ROOT2024
              LITERAL Float 3.14f64 42:Root[0000, 0]@0..100#ROOT2024
              LITERAL Float 3.14 42:Root[0000, 0]@0..100#ROOT2024
              PUNCH   - [alone] 42:Root[0000, 0]@0..100#ROOT2024
              LITERAL Float 3.14 42:Root[0000, 0]@0..100#ROOT2024
              LITERAL Integer 123i64 42:Root[0000, 0]@0..100#ROOT2024
              PUNCH   - [alone] 42:Root[0000, 0]@0..100#ROOT2024
              LITERAL Integer 123i64 42:Root[0000, 0]@0..100#ROOT2024
              LITERAL Integer 123 42:Root[0000, 0]@0..100#ROOT2024
              PUNCH   - [alone] 42:Root[0000, 0]@0..100#ROOT2024
              LITERAL Integer 123 42:Root[0000, 0]@0..100#ROOT2024"#]],
    );
}

#[test]
fn test_fn_like_mk_idents() {
    assert_expand(
        "fn_like_mk_idents",
        r#""#,
        expect![[r#"
            SUBTREE $$ 1 1



            SUBTREE $$ 1 1
              IDENT   standard 1
              IDENT   r#raw 1"#]],
        expect![[r#"
            SUBTREE $$ 42:Root[0000, 0]@0..100#ROOT2024 42:Root[0000, 0]@0..100#ROOT2024



            SUBTREE $$ 42:Root[0000, 0]@0..100#ROOT2024 42:Root[0000, 0]@0..100#ROOT2024
              IDENT   standard 42:Root[0000, 0]@0..100#ROOT2024
              IDENT   r#raw 42:Root[0000, 0]@0..100#ROOT2024"#]],
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
              LITERAL CStr null 1



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
            SUBTREE $$ 42:Root[0000, 0]@0..100#ROOT2024 42:Root[0000, 0]@0..100#ROOT2024
              LITERAL Integer 1u16 42:Root[0000, 0]@0..4#ROOT2024
              PUNCH   , [alone] 42:Root[0000, 0]@4..5#ROOT2024
              LITERAL Integer 2_u32 42:Root[0000, 0]@6..11#ROOT2024
              PUNCH   , [alone] 42:Root[0000, 0]@11..12#ROOT2024
              PUNCH   - [alone] 42:Root[0000, 0]@13..14#ROOT2024
              LITERAL Integer 4i64 42:Root[0000, 0]@14..18#ROOT2024
              PUNCH   , [alone] 42:Root[0000, 0]@18..19#ROOT2024
              LITERAL Float 3.14f32 42:Root[0000, 0]@20..27#ROOT2024
              PUNCH   , [alone] 42:Root[0000, 0]@27..28#ROOT2024
              LITERAL Str hello bridge 42:Root[0000, 0]@29..43#ROOT2024
              PUNCH   , [alone] 42:Root[0000, 0]@43..44#ROOT2024
              LITERAL Str suffixedsuffix 42:Root[0000, 0]@45..61#ROOT2024
              PUNCH   , [alone] 42:Root[0000, 0]@61..62#ROOT2024
              LITERAL StrRaw(2) raw 42:Root[0000, 0]@63..73#ROOT2024
              PUNCH   , [alone] 42:Root[0000, 0]@73..74#ROOT2024
              LITERAL Char a 42:Root[0000, 0]@75..78#ROOT2024
              PUNCH   , [alone] 42:Root[0000, 0]@78..79#ROOT2024
              LITERAL Byte b 42:Root[0000, 0]@80..84#ROOT2024
              PUNCH   , [alone] 42:Root[0000, 0]@84..85#ROOT2024
              LITERAL CStr null 42:Root[0000, 0]@86..93#ROOT2024



            SUBTREE $$ 42:Root[0000, 0]@0..100#ROOT2024 42:Root[0000, 0]@0..100#ROOT2024
              LITERAL Integer 1u16 42:Root[0000, 0]@0..4#ROOT2024
              PUNCH   , [alone] 42:Root[0000, 0]@4..5#ROOT2024
              LITERAL Integer 2_u32 42:Root[0000, 0]@6..11#ROOT2024
              PUNCH   , [alone] 42:Root[0000, 0]@11..12#ROOT2024
              PUNCH   - [alone] 42:Root[0000, 0]@13..14#ROOT2024
              LITERAL Integer 4i64 42:Root[0000, 0]@14..18#ROOT2024
              PUNCH   , [alone] 42:Root[0000, 0]@18..19#ROOT2024
              LITERAL Float 3.14f32 42:Root[0000, 0]@20..27#ROOT2024
              PUNCH   , [alone] 42:Root[0000, 0]@27..28#ROOT2024
              LITERAL Str hello bridge 42:Root[0000, 0]@29..43#ROOT2024
              PUNCH   , [alone] 42:Root[0000, 0]@43..44#ROOT2024
              LITERAL Str suffixedsuffix 42:Root[0000, 0]@45..61#ROOT2024
              PUNCH   , [alone] 42:Root[0000, 0]@61..62#ROOT2024
              LITERAL StrRaw(2) raw 42:Root[0000, 0]@63..73#ROOT2024
              PUNCH   , [alone] 42:Root[0000, 0]@73..74#ROOT2024
              LITERAL Char a 42:Root[0000, 0]@75..78#ROOT2024
              PUNCH   , [alone] 42:Root[0000, 0]@78..79#ROOT2024
              LITERAL Byte b 42:Root[0000, 0]@80..84#ROOT2024
              PUNCH   , [alone] 42:Root[0000, 0]@84..85#ROOT2024
              LITERAL CStr null 42:Root[0000, 0]@86..93#ROOT2024"#]],
    );
}

#[test]
fn test_fn_like_macro_negative_literals() {
    assert_expand(
        "fn_like_clone_tokens",
        r###"-1u16, - 2_u32, -3.14f32, - 2.7"###,
        expect![[r#"
            SUBTREE $$ 1 1
              PUNCH   - [alone] 1
              LITERAL Integer 1u16 1
              PUNCH   , [alone] 1
              PUNCH   - [alone] 1
              LITERAL Integer 2_u32 1
              PUNCH   , [alone] 1
              PUNCH   - [alone] 1
              LITERAL Float 3.14f32 1
              PUNCH   , [alone] 1
              PUNCH   - [alone] 1
              LITERAL Float 2.7 1



            SUBTREE $$ 1 1
              PUNCH   - [alone] 1
              LITERAL Integer 1u16 1
              PUNCH   , [alone] 1
              PUNCH   - [alone] 1
              LITERAL Integer 2_u32 1
              PUNCH   , [alone] 1
              PUNCH   - [alone] 1
              LITERAL Float 3.14f32 1
              PUNCH   , [alone] 1
              PUNCH   - [alone] 1
              LITERAL Float 2.7 1"#]],
        expect![[r#"
            SUBTREE $$ 42:Root[0000, 0]@0..100#ROOT2024 42:Root[0000, 0]@0..100#ROOT2024
              PUNCH   - [alone] 42:Root[0000, 0]@0..1#ROOT2024
              LITERAL Integer 1u16 42:Root[0000, 0]@1..5#ROOT2024
              PUNCH   , [alone] 42:Root[0000, 0]@5..6#ROOT2024
              PUNCH   - [alone] 42:Root[0000, 0]@7..8#ROOT2024
              LITERAL Integer 2_u32 42:Root[0000, 0]@9..14#ROOT2024
              PUNCH   , [alone] 42:Root[0000, 0]@14..15#ROOT2024
              PUNCH   - [alone] 42:Root[0000, 0]@16..17#ROOT2024
              LITERAL Float 3.14f32 42:Root[0000, 0]@17..24#ROOT2024
              PUNCH   , [alone] 42:Root[0000, 0]@24..25#ROOT2024
              PUNCH   - [alone] 42:Root[0000, 0]@26..27#ROOT2024
              LITERAL Float 2.7 42:Root[0000, 0]@28..31#ROOT2024



            SUBTREE $$ 42:Root[0000, 0]@0..100#ROOT2024 42:Root[0000, 0]@0..100#ROOT2024
              PUNCH   - [alone] 42:Root[0000, 0]@0..1#ROOT2024
              LITERAL Integer 1u16 42:Root[0000, 0]@1..5#ROOT2024
              PUNCH   , [alone] 42:Root[0000, 0]@5..6#ROOT2024
              PUNCH   - [alone] 42:Root[0000, 0]@7..8#ROOT2024
              LITERAL Integer 2_u32 42:Root[0000, 0]@9..14#ROOT2024
              PUNCH   , [alone] 42:Root[0000, 0]@14..15#ROOT2024
              PUNCH   - [alone] 42:Root[0000, 0]@16..17#ROOT2024
              LITERAL Float 3.14f32 42:Root[0000, 0]@17..24#ROOT2024
              PUNCH   , [alone] 42:Root[0000, 0]@24..25#ROOT2024
              PUNCH   - [alone] 42:Root[0000, 0]@26..27#ROOT2024
              LITERAL Float 2.7 42:Root[0000, 0]@28..31#ROOT2024"#]],
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
              IDENT   mod 1
              IDENT   m 1
              SUBTREE {} 1 1

            SUBTREE $$ 1 1
              IDENT   some 1
              IDENT   arguments 1

            SUBTREE $$ 1 1
              IDENT   compile_error 1
              PUNCH   ! [alone] 1
              SUBTREE () 1 1
                LITERAL Str #[attr_error(some arguments)] mod m {} 1
              PUNCH   ; [alone] 1"#]],
        expect![[r#"
            SUBTREE $$ 42:Root[0000, 0]@0..100#ROOT2024 42:Root[0000, 0]@0..100#ROOT2024
              IDENT   mod 42:Root[0000, 0]@0..3#ROOT2024
              IDENT   m 42:Root[0000, 0]@4..5#ROOT2024
              SUBTREE {} 42:Root[0000, 0]@6..7#ROOT2024 42:Root[0000, 0]@7..8#ROOT2024

            SUBTREE $$ 42:Root[0000, 0]@0..100#ROOT2024 42:Root[0000, 0]@0..100#ROOT2024
              IDENT   some 42:Root[0000, 0]@0..4#ROOT2024
              IDENT   arguments 42:Root[0000, 0]@5..14#ROOT2024

            SUBTREE $$ 42:Root[0000, 0]@0..100#ROOT2024 42:Root[0000, 0]@0..100#ROOT2024
              IDENT   compile_error 42:Root[0000, 0]@0..100#ROOT2024
              PUNCH   ! [alone] 42:Root[0000, 0]@0..100#ROOT2024
              SUBTREE () 42:Root[0000, 0]@0..100#ROOT2024 42:Root[0000, 0]@0..100#ROOT2024
                LITERAL Str #[attr_error(some arguments)] mod m {} 42:Root[0000, 0]@0..100#ROOT2024
              PUNCH   ; [alone] 42:Root[0000, 0]@0..100#ROOT2024"#]],
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
