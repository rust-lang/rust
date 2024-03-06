//! proc-macro tests

#[macro_use]
mod utils;
use utils::*;

use expect_test::expect;

#[test]
fn test_derive_empty() {
    assert_expand("DeriveEmpty", r#"struct S;"#, expect!["SUBTREE $$ 1 1"], expect!["SUBTREE $$ SpanData { range: 0..100, anchor: SpanAnchor(FileId(42), 2), ctx: SyntaxContextId(0) } SpanData { range: 0..100, anchor: SpanAnchor(FileId(42), 2), ctx: SyntaxContextId(0) }"]);
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
        expect![[r##"
            SUBTREE $$ SpanData { range: 0..100, anchor: SpanAnchor(FileId(42), 2), ctx: SyntaxContextId(0) } SpanData { range: 0..100, anchor: SpanAnchor(FileId(42), 2), ctx: SyntaxContextId(0) }
              IDENT   compile_error SpanData { range: 0..100, anchor: SpanAnchor(FileId(42), 2), ctx: SyntaxContextId(0) }
              PUNCH   ! [alone] SpanData { range: 0..100, anchor: SpanAnchor(FileId(42), 2), ctx: SyntaxContextId(0) }
              SUBTREE () SpanData { range: 0..100, anchor: SpanAnchor(FileId(42), 2), ctx: SyntaxContextId(0) } SpanData { range: 0..100, anchor: SpanAnchor(FileId(42), 2), ctx: SyntaxContextId(0) }
                LITERAL "#[derive(DeriveError)] struct S ;" SpanData { range: 0..100, anchor: SpanAnchor(FileId(42), 2), ctx: SyntaxContextId(0) }
              PUNCH   ; [alone] SpanData { range: 0..100, anchor: SpanAnchor(FileId(42), 2), ctx: SyntaxContextId(0) }"##]],
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
        expect![[r#"
            SUBTREE $$ SpanData { range: 0..100, anchor: SpanAnchor(FileId(42), 2), ctx: SyntaxContextId(0) } SpanData { range: 0..100, anchor: SpanAnchor(FileId(42), 2), ctx: SyntaxContextId(0) }
              IDENT   ident SpanData { range: 0..5, anchor: SpanAnchor(FileId(42), 2), ctx: SyntaxContextId(0) }
              PUNCH   , [alone] SpanData { range: 5..6, anchor: SpanAnchor(FileId(42), 2), ctx: SyntaxContextId(0) }
              LITERAL 0 SpanData { range: 7..8, anchor: SpanAnchor(FileId(42), 2), ctx: SyntaxContextId(0) }
              PUNCH   , [alone] SpanData { range: 8..9, anchor: SpanAnchor(FileId(42), 2), ctx: SyntaxContextId(0) }
              LITERAL 1 SpanData { range: 10..11, anchor: SpanAnchor(FileId(42), 2), ctx: SyntaxContextId(0) }
              PUNCH   , [alone] SpanData { range: 11..12, anchor: SpanAnchor(FileId(42), 2), ctx: SyntaxContextId(0) }
              SUBTREE [] SpanData { range: 13..14, anchor: SpanAnchor(FileId(42), 2), ctx: SyntaxContextId(0) } SpanData { range: 14..15, anchor: SpanAnchor(FileId(42), 2), ctx: SyntaxContextId(0) }"#]],
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
            SUBTREE $$ SpanData { range: 0..100, anchor: SpanAnchor(FileId(42), 2), ctx: SyntaxContextId(0) } SpanData { range: 0..100, anchor: SpanAnchor(FileId(42), 2), ctx: SyntaxContextId(0) }
              IDENT   ident SpanData { range: 0..5, anchor: SpanAnchor(FileId(42), 2), ctx: SyntaxContextId(0) }
              PUNCH   , [alone] SpanData { range: 5..6, anchor: SpanAnchor(FileId(42), 2), ctx: SyntaxContextId(0) }
              SUBTREE [] SpanData { range: 7..8, anchor: SpanAnchor(FileId(42), 2), ctx: SyntaxContextId(0) } SpanData { range: 7..8, anchor: SpanAnchor(FileId(42), 2), ctx: SyntaxContextId(0) }"#]],
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
            SUBTREE $$ SpanData { range: 0..100, anchor: SpanAnchor(FileId(42), 2), ctx: SyntaxContextId(0) } SpanData { range: 0..100, anchor: SpanAnchor(FileId(42), 2), ctx: SyntaxContextId(0) }
              IDENT   r#async SpanData { range: 0..7, anchor: SpanAnchor(FileId(42), 2), ctx: SyntaxContextId(0) }"#]],
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
            SUBTREE $$ SpanData { range: 0..100, anchor: SpanAnchor(FileId(42), 2), ctx: SyntaxContextId(0) } SpanData { range: 0..100, anchor: SpanAnchor(FileId(42), 2), ctx: SyntaxContextId(0) }
              IDENT   r#joined SpanData { range: 0..11, anchor: SpanAnchor(FileId(42), 2), ctx: SyntaxContextId(0) }"#]],
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
            SUBTREE $$ SpanData { range: 0..100, anchor: SpanAnchor(FileId(42), 2), ctx: SyntaxContextId(0) } SpanData { range: 0..100, anchor: SpanAnchor(FileId(42), 2), ctx: SyntaxContextId(0) }
              IDENT   set_def_site SpanData { range: 0..150, anchor: SpanAnchor(FileId(41), 1), ctx: SyntaxContextId(0) }
              IDENT   resolved_at_def_site SpanData { range: 13..33, anchor: SpanAnchor(FileId(42), 2), ctx: SyntaxContextId(0) }
              IDENT   start_span SpanData { range: 34..34, anchor: SpanAnchor(FileId(42), 2), ctx: SyntaxContextId(0) }"#]],
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
        expect![[r#"
            SUBTREE $$ SpanData { range: 0..100, anchor: SpanAnchor(FileId(42), 2), ctx: SyntaxContextId(0) } SpanData { range: 0..100, anchor: SpanAnchor(FileId(42), 2), ctx: SyntaxContextId(0) }
              LITERAL b"byte_string" SpanData { range: 0..100, anchor: SpanAnchor(FileId(42), 2), ctx: SyntaxContextId(0) }
              LITERAL 'c' SpanData { range: 0..100, anchor: SpanAnchor(FileId(42), 2), ctx: SyntaxContextId(0) }
              LITERAL "string" SpanData { range: 0..100, anchor: SpanAnchor(FileId(42), 2), ctx: SyntaxContextId(0) }
              LITERAL 3.14f64 SpanData { range: 0..100, anchor: SpanAnchor(FileId(42), 2), ctx: SyntaxContextId(0) }
              LITERAL 3.14 SpanData { range: 0..100, anchor: SpanAnchor(FileId(42), 2), ctx: SyntaxContextId(0) }
              LITERAL 123i64 SpanData { range: 0..100, anchor: SpanAnchor(FileId(42), 2), ctx: SyntaxContextId(0) }
              LITERAL 123 SpanData { range: 0..100, anchor: SpanAnchor(FileId(42), 2), ctx: SyntaxContextId(0) }"#]],
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
            SUBTREE $$ SpanData { range: 0..100, anchor: SpanAnchor(FileId(42), 2), ctx: SyntaxContextId(0) } SpanData { range: 0..100, anchor: SpanAnchor(FileId(42), 2), ctx: SyntaxContextId(0) }
              IDENT   standard SpanData { range: 0..100, anchor: SpanAnchor(FileId(42), 2), ctx: SyntaxContextId(0) }
              IDENT   r#raw SpanData { range: 0..100, anchor: SpanAnchor(FileId(42), 2), ctx: SyntaxContextId(0) }"#]],
    );
}

#[test]
fn test_fn_like_macro_clone_literals() {
    assert_expand(
        "fn_like_clone_tokens",
        r###"1u16, 2_u32, -4i64, 3.14f32, "hello bridge", "suffixed"suffix, r##"raw"##, 'a', b'b', c"null""###,
        expect![[r###"
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
              LITERAL "hello bridge" 1
              PUNCH   , [alone] 1
              LITERAL "suffixed"suffix 1
              PUNCH   , [alone] 1
              LITERAL r##"raw"## 1
              PUNCH   , [alone] 1
              LITERAL 'a' 1
              PUNCH   , [alone] 1
              LITERAL b'b' 1
              PUNCH   , [alone] 1
              LITERAL c"null" 1"###]],
        expect![[r###"
            SUBTREE $$ SpanData { range: 0..100, anchor: SpanAnchor(FileId(42), 2), ctx: SyntaxContextId(0) } SpanData { range: 0..100, anchor: SpanAnchor(FileId(42), 2), ctx: SyntaxContextId(0) }
              LITERAL 1u16 SpanData { range: 0..4, anchor: SpanAnchor(FileId(42), 2), ctx: SyntaxContextId(0) }
              PUNCH   , [alone] SpanData { range: 4..5, anchor: SpanAnchor(FileId(42), 2), ctx: SyntaxContextId(0) }
              LITERAL 2_u32 SpanData { range: 6..11, anchor: SpanAnchor(FileId(42), 2), ctx: SyntaxContextId(0) }
              PUNCH   , [alone] SpanData { range: 11..12, anchor: SpanAnchor(FileId(42), 2), ctx: SyntaxContextId(0) }
              PUNCH   - [alone] SpanData { range: 13..14, anchor: SpanAnchor(FileId(42), 2), ctx: SyntaxContextId(0) }
              LITERAL 4i64 SpanData { range: 14..18, anchor: SpanAnchor(FileId(42), 2), ctx: SyntaxContextId(0) }
              PUNCH   , [alone] SpanData { range: 18..19, anchor: SpanAnchor(FileId(42), 2), ctx: SyntaxContextId(0) }
              LITERAL 3.14f32 SpanData { range: 20..27, anchor: SpanAnchor(FileId(42), 2), ctx: SyntaxContextId(0) }
              PUNCH   , [alone] SpanData { range: 27..28, anchor: SpanAnchor(FileId(42), 2), ctx: SyntaxContextId(0) }
              LITERAL "hello bridge" SpanData { range: 29..43, anchor: SpanAnchor(FileId(42), 2), ctx: SyntaxContextId(0) }
              PUNCH   , [alone] SpanData { range: 43..44, anchor: SpanAnchor(FileId(42), 2), ctx: SyntaxContextId(0) }
              LITERAL "suffixed"suffix SpanData { range: 45..61, anchor: SpanAnchor(FileId(42), 2), ctx: SyntaxContextId(0) }
              PUNCH   , [alone] SpanData { range: 61..62, anchor: SpanAnchor(FileId(42), 2), ctx: SyntaxContextId(0) }
              LITERAL r##"raw"## SpanData { range: 63..73, anchor: SpanAnchor(FileId(42), 2), ctx: SyntaxContextId(0) }
              PUNCH   , [alone] SpanData { range: 73..74, anchor: SpanAnchor(FileId(42), 2), ctx: SyntaxContextId(0) }
              LITERAL 'a' SpanData { range: 75..78, anchor: SpanAnchor(FileId(42), 2), ctx: SyntaxContextId(0) }
              PUNCH   , [alone] SpanData { range: 78..79, anchor: SpanAnchor(FileId(42), 2), ctx: SyntaxContextId(0) }
              LITERAL b'b' SpanData { range: 80..84, anchor: SpanAnchor(FileId(42), 2), ctx: SyntaxContextId(0) }
              PUNCH   , [alone] SpanData { range: 84..85, anchor: SpanAnchor(FileId(42), 2), ctx: SyntaxContextId(0) }
              LITERAL c"null" SpanData { range: 86..93, anchor: SpanAnchor(FileId(42), 2), ctx: SyntaxContextId(0) }"###]],
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
        expect![[r##"
            SUBTREE $$ SpanData { range: 0..100, anchor: SpanAnchor(FileId(42), 2), ctx: SyntaxContextId(0) } SpanData { range: 0..100, anchor: SpanAnchor(FileId(42), 2), ctx: SyntaxContextId(0) }
              IDENT   compile_error SpanData { range: 0..100, anchor: SpanAnchor(FileId(42), 2), ctx: SyntaxContextId(0) }
              PUNCH   ! [alone] SpanData { range: 0..100, anchor: SpanAnchor(FileId(42), 2), ctx: SyntaxContextId(0) }
              SUBTREE () SpanData { range: 0..100, anchor: SpanAnchor(FileId(42), 2), ctx: SyntaxContextId(0) } SpanData { range: 0..100, anchor: SpanAnchor(FileId(42), 2), ctx: SyntaxContextId(0) }
                LITERAL "#[attr_error(some arguments)] mod m {}" SpanData { range: 0..100, anchor: SpanAnchor(FileId(42), 2), ctx: SyntaxContextId(0) }
              PUNCH   ; [alone] SpanData { range: 0..100, anchor: SpanAnchor(FileId(42), 2), ctx: SyntaxContextId(0) }"##]],
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
