//! proc-macro tests

#[macro_use]
mod utils;
use utils::*;

use expect_test::expect;

#[test]
fn test_derive_empty() {
    assert_expand(
        "DeriveEmpty",
        r#"struct S { field: &'r#lt fn(u32) -> &'a r#u32 }"#,
        expect![[r#"
            IDENT 1 struct
            IDENT 1 S
            GROUP {} 1 1 1
              IDENT 1 field
              PUNCT 1 : [alone]
              PUNCT 1 & [joint]
              PUNCT 1 ' [joint]
              IDENT 1 r#lt
              IDENT 1 fn
              GROUP () 1 1 1
                IDENT 1 u32
              PUNCT 1 - [joint]
              PUNCT 1 > [alone]
              PUNCT 1 & [joint]
              PUNCT 1 ' [joint]
              IDENT 1 a
              IDENT 1 r#u32
        "#]],
        expect![[r#"
            IDENT 42:Root[0000, 0]@0..6#ROOT2024 struct
            IDENT 42:Root[0000, 0]@7..8#ROOT2024 S
            GROUP {} 42:Root[0000, 0]@9..10#ROOT2024 42:Root[0000, 0]@46..47#ROOT2024 42:Root[0000, 0]@9..47#ROOT2024
              IDENT 42:Root[0000, 0]@11..16#ROOT2024 field
              PUNCT 42:Root[0000, 0]@16..17#ROOT2024 : [alone]
              PUNCT 42:Root[0000, 0]@18..19#ROOT2024 & [joint]
              PUNCT 42:Root[0000, 0]@22..23#ROOT2024 ' [joint]
              IDENT 42:Root[0000, 0]@22..24#ROOT2024 r#lt
              IDENT 42:Root[0000, 0]@25..27#ROOT2024 fn
              GROUP () 42:Root[0000, 0]@27..28#ROOT2024 42:Root[0000, 0]@31..32#ROOT2024 42:Root[0000, 0]@27..32#ROOT2024
                IDENT 42:Root[0000, 0]@28..31#ROOT2024 u32
              PUNCT 42:Root[0000, 0]@33..34#ROOT2024 - [joint]
              PUNCT 42:Root[0000, 0]@34..35#ROOT2024 > [alone]
              PUNCT 42:Root[0000, 0]@36..37#ROOT2024 & [joint]
              PUNCT 42:Root[0000, 0]@38..39#ROOT2024 ' [joint]
              IDENT 42:Root[0000, 0]@38..39#ROOT2024 a
              IDENT 42:Root[0000, 0]@42..45#ROOT2024 r#u32
        "#]],
    );
}

#[test]
fn test_derive_reemit_helpers() {
    assert_expand(
        "DeriveReemit",
        r#"
#[helper(build_fn(private, name = "partial_build"))]
pub struct Foo {
    /// The domain where this federated instance is running
    #[helper(setter(into))]
    pub(crate) domain: String,
}
"#,
        expect![[r#"
            PUNCT 1 # [joint]
            GROUP [] 1 1 1
              IDENT 1 helper
              GROUP () 1 1 1
                IDENT 1 build_fn
                GROUP () 1 1 1
                  IDENT 1 private
                  PUNCT 1 , [alone]
                  IDENT 1 name
                  PUNCT 1 = [alone]
                  LITER 1 Str partial_build
            IDENT 1 pub
            IDENT 1 struct
            IDENT 1 Foo
            GROUP {} 1 1 1
              PUNCT 1 # [alone]
              GROUP [] 1 1 1
                IDENT 1 doc
                PUNCT 1 = [alone]
                LITER 1 Str / The domain where this federated instance is running
              PUNCT 1 # [joint]
              GROUP [] 1 1 1
                IDENT 1 helper
                GROUP () 1 1 1
                  IDENT 1 setter
                  GROUP () 1 1 1
                    IDENT 1 into
              IDENT 1 pub
              GROUP () 1 1 1
                IDENT 1 crate
              IDENT 1 domain
              PUNCT 1 : [alone]
              IDENT 1 String
              PUNCT 1 , [alone]


            PUNCT 1 # [joint]
            GROUP [] 1 1 1
              IDENT 1 helper
              GROUP () 1 1 1
                IDENT 1 build_fn
                GROUP () 1 1 1
                  IDENT 1 private
                  PUNCT 1 , [alone]
                  IDENT 1 name
                  PUNCT 1 = [alone]
                  LITER 1 Str partial_build
            IDENT 1 pub
            IDENT 1 struct
            IDENT 1 Foo
            GROUP {} 1 1 1
              PUNCT 1 # [alone]
              GROUP [] 1 1 1
                IDENT 1 doc
                PUNCT 1 = [alone]
                LITER 1 Str / The domain where this federated instance is running
              PUNCT 1 # [joint]
              GROUP [] 1 1 1
                IDENT 1 helper
                GROUP () 1 1 1
                  IDENT 1 setter
                  GROUP () 1 1 1
                    IDENT 1 into
              IDENT 1 pub
              GROUP () 1 1 1
                IDENT 1 crate
              IDENT 1 domain
              PUNCT 1 : [alone]
              IDENT 1 String
              PUNCT 1 , [alone]
        "#]],
        expect![[r#"
            PUNCT 42:Root[0000, 0]@1..2#ROOT2024 # [joint]
            GROUP [] 42:Root[0000, 0]@2..3#ROOT2024 42:Root[0000, 0]@52..53#ROOT2024 42:Root[0000, 0]@2..53#ROOT2024
              IDENT 42:Root[0000, 0]@3..9#ROOT2024 helper
              GROUP () 42:Root[0000, 0]@9..10#ROOT2024 42:Root[0000, 0]@51..52#ROOT2024 42:Root[0000, 0]@9..52#ROOT2024
                IDENT 42:Root[0000, 0]@10..18#ROOT2024 build_fn
                GROUP () 42:Root[0000, 0]@18..19#ROOT2024 42:Root[0000, 0]@50..51#ROOT2024 42:Root[0000, 0]@18..51#ROOT2024
                  IDENT 42:Root[0000, 0]@19..26#ROOT2024 private
                  PUNCT 42:Root[0000, 0]@26..27#ROOT2024 , [alone]
                  IDENT 42:Root[0000, 0]@28..32#ROOT2024 name
                  PUNCT 42:Root[0000, 0]@33..34#ROOT2024 = [alone]
                  LITER 42:Root[0000, 0]@35..50#ROOT2024 Str partial_build
            IDENT 42:Root[0000, 0]@54..57#ROOT2024 pub
            IDENT 42:Root[0000, 0]@58..64#ROOT2024 struct
            IDENT 42:Root[0000, 0]@65..68#ROOT2024 Foo
            GROUP {} 42:Root[0000, 0]@69..70#ROOT2024 42:Root[0000, 0]@190..191#ROOT2024 42:Root[0000, 0]@69..191#ROOT2024
              PUNCT 42:Root[0000, 0]@0..0#ROOT2024 # [alone]
              GROUP [] 42:Root[0000, 0]@0..0#ROOT2024 42:Root[0000, 0]@0..0#ROOT2024 42:Root[0000, 0]@0..0#ROOT2024
                IDENT 42:Root[0000, 0]@0..0#ROOT2024 doc
                PUNCT 42:Root[0000, 0]@0..0#ROOT2024 = [alone]
                LITER 42:Root[0000, 0]@75..130#ROOT2024 Str / The domain where this federated instance is running
              PUNCT 42:Root[0000, 0]@135..136#ROOT2024 # [joint]
              GROUP [] 42:Root[0000, 0]@136..137#ROOT2024 42:Root[0000, 0]@157..158#ROOT2024 42:Root[0000, 0]@136..158#ROOT2024
                IDENT 42:Root[0000, 0]@137..143#ROOT2024 helper
                GROUP () 42:Root[0000, 0]@143..144#ROOT2024 42:Root[0000, 0]@156..157#ROOT2024 42:Root[0000, 0]@143..157#ROOT2024
                  IDENT 42:Root[0000, 0]@144..150#ROOT2024 setter
                  GROUP () 42:Root[0000, 0]@150..151#ROOT2024 42:Root[0000, 0]@155..156#ROOT2024 42:Root[0000, 0]@150..156#ROOT2024
                    IDENT 42:Root[0000, 0]@151..155#ROOT2024 into
              IDENT 42:Root[0000, 0]@163..166#ROOT2024 pub
              GROUP () 42:Root[0000, 0]@166..167#ROOT2024 42:Root[0000, 0]@172..173#ROOT2024 42:Root[0000, 0]@166..173#ROOT2024
                IDENT 42:Root[0000, 0]@167..172#ROOT2024 crate
              IDENT 42:Root[0000, 0]@174..180#ROOT2024 domain
              PUNCT 42:Root[0000, 0]@180..181#ROOT2024 : [alone]
              IDENT 42:Root[0000, 0]@182..188#ROOT2024 String
              PUNCT 42:Root[0000, 0]@188..189#ROOT2024 , [alone]


            PUNCT 42:Root[0000, 0]@1..2#ROOT2024 # [joint]
            GROUP [] 42:Root[0000, 0]@2..3#ROOT2024 42:Root[0000, 0]@52..53#ROOT2024 42:Root[0000, 0]@2..53#ROOT2024
              IDENT 42:Root[0000, 0]@3..9#ROOT2024 helper
              GROUP () 42:Root[0000, 0]@9..10#ROOT2024 42:Root[0000, 0]@51..52#ROOT2024 42:Root[0000, 0]@9..52#ROOT2024
                IDENT 42:Root[0000, 0]@10..18#ROOT2024 build_fn
                GROUP () 42:Root[0000, 0]@18..19#ROOT2024 42:Root[0000, 0]@50..51#ROOT2024 42:Root[0000, 0]@18..51#ROOT2024
                  IDENT 42:Root[0000, 0]@19..26#ROOT2024 private
                  PUNCT 42:Root[0000, 0]@26..27#ROOT2024 , [alone]
                  IDENT 42:Root[0000, 0]@28..32#ROOT2024 name
                  PUNCT 42:Root[0000, 0]@33..34#ROOT2024 = [alone]
                  LITER 42:Root[0000, 0]@35..50#ROOT2024 Str partial_build
            IDENT 42:Root[0000, 0]@54..57#ROOT2024 pub
            IDENT 42:Root[0000, 0]@58..64#ROOT2024 struct
            IDENT 42:Root[0000, 0]@65..68#ROOT2024 Foo
            GROUP {} 42:Root[0000, 0]@69..70#ROOT2024 42:Root[0000, 0]@190..191#ROOT2024 42:Root[0000, 0]@69..191#ROOT2024
              PUNCT 42:Root[0000, 0]@0..0#ROOT2024 # [alone]
              GROUP [] 42:Root[0000, 0]@0..0#ROOT2024 42:Root[0000, 0]@0..0#ROOT2024 42:Root[0000, 0]@0..0#ROOT2024
                IDENT 42:Root[0000, 0]@0..0#ROOT2024 doc
                PUNCT 42:Root[0000, 0]@0..0#ROOT2024 = [alone]
                LITER 42:Root[0000, 0]@75..130#ROOT2024 Str / The domain where this federated instance is running
              PUNCT 42:Root[0000, 0]@135..136#ROOT2024 # [joint]
              GROUP [] 42:Root[0000, 0]@136..137#ROOT2024 42:Root[0000, 0]@157..158#ROOT2024 42:Root[0000, 0]@136..158#ROOT2024
                IDENT 42:Root[0000, 0]@137..143#ROOT2024 helper
                GROUP () 42:Root[0000, 0]@143..144#ROOT2024 42:Root[0000, 0]@156..157#ROOT2024 42:Root[0000, 0]@143..157#ROOT2024
                  IDENT 42:Root[0000, 0]@144..150#ROOT2024 setter
                  GROUP () 42:Root[0000, 0]@150..151#ROOT2024 42:Root[0000, 0]@155..156#ROOT2024 42:Root[0000, 0]@150..156#ROOT2024
                    IDENT 42:Root[0000, 0]@151..155#ROOT2024 into
              IDENT 42:Root[0000, 0]@163..166#ROOT2024 pub
              GROUP () 42:Root[0000, 0]@166..167#ROOT2024 42:Root[0000, 0]@172..173#ROOT2024 42:Root[0000, 0]@166..173#ROOT2024
                IDENT 42:Root[0000, 0]@167..172#ROOT2024 crate
              IDENT 42:Root[0000, 0]@174..180#ROOT2024 domain
              PUNCT 42:Root[0000, 0]@180..181#ROOT2024 : [alone]
              IDENT 42:Root[0000, 0]@182..188#ROOT2024 String
              PUNCT 42:Root[0000, 0]@188..189#ROOT2024 , [alone]
        "#]],
    );
}

#[test]
fn test_derive_error() {
    assert_expand(
        "DeriveError",
        r#"struct S { field: u32 }"#,
        expect![[r#"
            IDENT 1 struct
            IDENT 1 S
            GROUP {} 1 1 1
              IDENT 1 field
              PUNCT 1 : [alone]
              IDENT 1 u32


            IDENT 1 compile_error
            PUNCT 1 ! [joint]
            GROUP () 1 1 1
              LITER 1 Str #[derive(DeriveError)] struct S {field : u32}
            PUNCT 1 ; [alone]
        "#]],
        expect![[r#"
            IDENT 42:Root[0000, 0]@0..6#ROOT2024 struct
            IDENT 42:Root[0000, 0]@7..8#ROOT2024 S
            GROUP {} 42:Root[0000, 0]@9..10#ROOT2024 42:Root[0000, 0]@22..23#ROOT2024 42:Root[0000, 0]@9..23#ROOT2024
              IDENT 42:Root[0000, 0]@11..16#ROOT2024 field
              PUNCT 42:Root[0000, 0]@16..17#ROOT2024 : [alone]
              IDENT 42:Root[0000, 0]@18..21#ROOT2024 u32


            IDENT 42:Root[0000, 0]@0..13#ROOT2024 compile_error
            PUNCT 42:Root[0000, 0]@13..14#ROOT2024 ! [joint]
            GROUP () 42:Root[0000, 0]@14..15#ROOT2024 42:Root[0000, 0]@62..63#ROOT2024 42:Root[0000, 0]@14..63#ROOT2024
              LITER 42:Root[0000, 0]@15..62#ROOT2024 Str #[derive(DeriveError)] struct S {field : u32}
            PUNCT 42:Root[0000, 0]@63..64#ROOT2024 ; [alone]
        "#]],
    );
}

#[test]
fn test_fn_like_macro_noop() {
    assert_expand(
        "fn_like_noop",
        r#"ident, 0, 1, []"#,
        expect![[r#"
            IDENT 1 ident
            PUNCT 1 , [alone]
            LITER 1 Integer 0
            PUNCT 1 , [alone]
            LITER 1 Integer 1
            PUNCT 1 , [alone]
            GROUP [] 1 1 1


            IDENT 1 ident
            PUNCT 1 , [alone]
            LITER 1 Integer 0
            PUNCT 1 , [alone]
            LITER 1 Integer 1
            PUNCT 1 , [alone]
            GROUP [] 1 1 1
        "#]],
        expect![[r#"
            IDENT 42:Root[0000, 0]@0..5#ROOT2024 ident
            PUNCT 42:Root[0000, 0]@5..6#ROOT2024 , [alone]
            LITER 42:Root[0000, 0]@7..8#ROOT2024 Integer 0
            PUNCT 42:Root[0000, 0]@8..9#ROOT2024 , [alone]
            LITER 42:Root[0000, 0]@10..11#ROOT2024 Integer 1
            PUNCT 42:Root[0000, 0]@11..12#ROOT2024 , [alone]
            GROUP [] 42:Root[0000, 0]@13..14#ROOT2024 42:Root[0000, 0]@14..15#ROOT2024 42:Root[0000, 0]@13..15#ROOT2024


            IDENT 42:Root[0000, 0]@0..5#ROOT2024 ident
            PUNCT 42:Root[0000, 0]@5..6#ROOT2024 , [alone]
            LITER 42:Root[0000, 0]@7..8#ROOT2024 Integer 0
            PUNCT 42:Root[0000, 0]@8..9#ROOT2024 , [alone]
            LITER 42:Root[0000, 0]@10..11#ROOT2024 Integer 1
            PUNCT 42:Root[0000, 0]@11..12#ROOT2024 , [alone]
            GROUP [] 42:Root[0000, 0]@13..14#ROOT2024 42:Root[0000, 0]@14..15#ROOT2024 42:Root[0000, 0]@13..15#ROOT2024
        "#]],
    );
}

#[test]
fn test_fn_like_macro_clone_ident_subtree() {
    assert_expand(
        "fn_like_clone_tokens",
        r#"ident, [ident2, ident3]"#,
        expect![[r#"
            IDENT 1 ident
            PUNCT 1 , [alone]
            GROUP [] 1 1 1
              IDENT 1 ident2
              PUNCT 1 , [alone]
              IDENT 1 ident3


            IDENT 1 ident
            PUNCT 1 , [alone]
            GROUP [] 1 1 1
              IDENT 1 ident2
              PUNCT 1 , [alone]
              IDENT 1 ident3
        "#]],
        expect![[r#"
            IDENT 42:Root[0000, 0]@0..5#ROOT2024 ident
            PUNCT 42:Root[0000, 0]@5..6#ROOT2024 , [alone]
            GROUP [] 42:Root[0000, 0]@7..8#ROOT2024 42:Root[0000, 0]@22..23#ROOT2024 42:Root[0000, 0]@7..23#ROOT2024
              IDENT 42:Root[0000, 0]@8..14#ROOT2024 ident2
              PUNCT 42:Root[0000, 0]@14..15#ROOT2024 , [alone]
              IDENT 42:Root[0000, 0]@16..22#ROOT2024 ident3


            IDENT 42:Root[0000, 0]@0..5#ROOT2024 ident
            PUNCT 42:Root[0000, 0]@5..6#ROOT2024 , [alone]
            GROUP [] 42:Root[0000, 0]@7..23#ROOT2024 42:Root[0000, 0]@7..23#ROOT2024 42:Root[0000, 0]@7..23#ROOT2024
              IDENT 42:Root[0000, 0]@8..14#ROOT2024 ident2
              PUNCT 42:Root[0000, 0]@14..15#ROOT2024 , [alone]
              IDENT 42:Root[0000, 0]@16..22#ROOT2024 ident3
        "#]],
    );
}

#[test]
fn test_fn_like_macro_clone_raw_ident() {
    assert_expand(
        "fn_like_clone_tokens",
        "r#async",
        expect![[r#"
            IDENT 1 r#async


            IDENT 1 r#async
        "#]],
        expect![[r#"
            IDENT 42:Root[0000, 0]@2..7#ROOT2024 r#async


            IDENT 42:Root[0000, 0]@2..7#ROOT2024 r#async
        "#]],
    );
}

#[test]
fn test_fn_like_fn_like_span_join() {
    assert_expand(
        "fn_like_span_join",
        "foo     bar",
        expect![[r#"
            IDENT 1 foo
            IDENT 1 bar


            IDENT 1 r#joined
        "#]],
        expect![[r#"
            IDENT 42:Root[0000, 0]@0..3#ROOT2024 foo
            IDENT 42:Root[0000, 0]@8..11#ROOT2024 bar


            IDENT 42:Root[0000, 0]@0..11#ROOT2024 r#joined
        "#]],
    );
}

#[test]
fn test_fn_like_fn_like_span_ops() {
    assert_expand(
        "fn_like_span_ops",
        "set_def_site resolved_at_def_site start_span",
        expect![[r#"
            IDENT 1 set_def_site
            IDENT 1 resolved_at_def_site
            IDENT 1 start_span


            IDENT 0 set_def_site
            IDENT 1 resolved_at_def_site
            IDENT 1 start_span
        "#]],
        expect![[r#"
            IDENT 42:Root[0000, 0]@0..12#ROOT2024 set_def_site
            IDENT 42:Root[0000, 0]@13..33#ROOT2024 resolved_at_def_site
            IDENT 42:Root[0000, 0]@34..44#ROOT2024 start_span


            IDENT 41:Root[0000, 0]@0..150#ROOT2024 set_def_site
            IDENT 42:Root[0000, 0]@13..33#ROOT2024 resolved_at_def_site
            IDENT 42:Root[0000, 0]@34..34#ROOT2024 start_span
        "#]],
    );
}

#[test]
fn test_fn_like_mk_literals() {
    assert_expand(
        "fn_like_mk_literals",
        r#""#,
        expect![[r#"


            LITER 1 ByteStr byte_string
            LITER 1 Char c
            LITER 1 Str string
            LITER 1 Str -string
            LITER 1 CStr cstring
            LITER 1 Float 3.14f64
            LITER 1 Float -3.14f64
            LITER 1 Float 3.14
            LITER 1 Float -3.14
            LITER 1 Integer 123i64
            LITER 1 Integer -123i64
            LITER 1 Integer 123
            LITER 1 Integer -123
        "#]],
        expect![[r#"


            LITER 42:Root[0000, 0]@0..100#ROOT2024 ByteStr byte_string
            LITER 42:Root[0000, 0]@0..100#ROOT2024 Char c
            LITER 42:Root[0000, 0]@0..100#ROOT2024 Str string
            LITER 42:Root[0000, 0]@0..100#ROOT2024 Str -string
            LITER 42:Root[0000, 0]@0..100#ROOT2024 CStr cstring
            LITER 42:Root[0000, 0]@0..100#ROOT2024 Float 3.14f64
            LITER 42:Root[0000, 0]@0..100#ROOT2024 Float -3.14f64
            LITER 42:Root[0000, 0]@0..100#ROOT2024 Float 3.14
            LITER 42:Root[0000, 0]@0..100#ROOT2024 Float -3.14
            LITER 42:Root[0000, 0]@0..100#ROOT2024 Integer 123i64
            LITER 42:Root[0000, 0]@0..100#ROOT2024 Integer -123i64
            LITER 42:Root[0000, 0]@0..100#ROOT2024 Integer 123
            LITER 42:Root[0000, 0]@0..100#ROOT2024 Integer -123
        "#]],
    );
}

#[test]
fn test_fn_like_mk_idents() {
    assert_expand(
        "fn_like_mk_idents",
        r#""#,
        expect![[r#"


            IDENT 1 standard
            IDENT 1 r#raw
        "#]],
        expect![[r#"


            IDENT 42:Root[0000, 0]@0..100#ROOT2024 standard
            IDENT 42:Root[0000, 0]@0..100#ROOT2024 r#raw
        "#]],
    );
}

#[test]
fn test_fn_like_macro_clone_literals() {
    assert_expand(
        "fn_like_clone_tokens",
        r###"1u16, 2_u32, -4i64, 3.14f32, "hello bridge", "suffixed"suffix, r##"raw"##, 'a', b'b', c"null""###,
        expect![[r#"
            LITER 1 Integer 1u16
            PUNCT 1 , [alone]
            LITER 1 Integer 2_u32
            PUNCT 1 , [alone]
            PUNCT 1 - [alone]
            LITER 1 Integer 4i64
            PUNCT 1 , [alone]
            LITER 1 Float 3.14f32
            PUNCT 1 , [alone]
            LITER 1 Str hello bridge
            PUNCT 1 , [alone]
            LITER 1 Str suffixedsuffix
            PUNCT 1 , [alone]
            LITER 1 StrRaw(2) raw
            PUNCT 1 , [alone]
            LITER 1 Char a
            PUNCT 1 , [alone]
            LITER 1 Byte b
            PUNCT 1 , [alone]
            LITER 1 CStr null


            LITER 1 Integer 1u16
            PUNCT 1 , [alone]
            LITER 1 Integer 2_u32
            PUNCT 1 , [alone]
            PUNCT 1 - [alone]
            LITER 1 Integer 4i64
            PUNCT 1 , [alone]
            LITER 1 Float 3.14f32
            PUNCT 1 , [alone]
            LITER 1 Str hello bridge
            PUNCT 1 , [alone]
            LITER 1 Str suffixedsuffix
            PUNCT 1 , [alone]
            LITER 1 StrRaw(2) raw
            PUNCT 1 , [alone]
            LITER 1 Char a
            PUNCT 1 , [alone]
            LITER 1 Byte b
            PUNCT 1 , [alone]
            LITER 1 CStr null
        "#]],
        expect![[r#"
            LITER 42:Root[0000, 0]@0..4#ROOT2024 Integer 1u16
            PUNCT 42:Root[0000, 0]@4..5#ROOT2024 , [alone]
            LITER 42:Root[0000, 0]@6..11#ROOT2024 Integer 2_u32
            PUNCT 42:Root[0000, 0]@11..12#ROOT2024 , [alone]
            PUNCT 42:Root[0000, 0]@13..14#ROOT2024 - [alone]
            LITER 42:Root[0000, 0]@14..18#ROOT2024 Integer 4i64
            PUNCT 42:Root[0000, 0]@18..19#ROOT2024 , [alone]
            LITER 42:Root[0000, 0]@20..27#ROOT2024 Float 3.14f32
            PUNCT 42:Root[0000, 0]@27..28#ROOT2024 , [alone]
            LITER 42:Root[0000, 0]@29..43#ROOT2024 Str hello bridge
            PUNCT 42:Root[0000, 0]@43..44#ROOT2024 , [alone]
            LITER 42:Root[0000, 0]@45..61#ROOT2024 Str suffixedsuffix
            PUNCT 42:Root[0000, 0]@61..62#ROOT2024 , [alone]
            LITER 42:Root[0000, 0]@63..73#ROOT2024 StrRaw(2) raw
            PUNCT 42:Root[0000, 0]@73..74#ROOT2024 , [alone]
            LITER 42:Root[0000, 0]@75..78#ROOT2024 Char a
            PUNCT 42:Root[0000, 0]@78..79#ROOT2024 , [alone]
            LITER 42:Root[0000, 0]@80..84#ROOT2024 Byte b
            PUNCT 42:Root[0000, 0]@84..85#ROOT2024 , [alone]
            LITER 42:Root[0000, 0]@86..93#ROOT2024 CStr null


            LITER 42:Root[0000, 0]@0..4#ROOT2024 Integer 1u16
            PUNCT 42:Root[0000, 0]@4..5#ROOT2024 , [alone]
            LITER 42:Root[0000, 0]@6..11#ROOT2024 Integer 2_u32
            PUNCT 42:Root[0000, 0]@11..12#ROOT2024 , [alone]
            PUNCT 42:Root[0000, 0]@13..14#ROOT2024 - [alone]
            LITER 42:Root[0000, 0]@14..18#ROOT2024 Integer 4i64
            PUNCT 42:Root[0000, 0]@18..19#ROOT2024 , [alone]
            LITER 42:Root[0000, 0]@20..27#ROOT2024 Float 3.14f32
            PUNCT 42:Root[0000, 0]@27..28#ROOT2024 , [alone]
            LITER 42:Root[0000, 0]@29..43#ROOT2024 Str hello bridge
            PUNCT 42:Root[0000, 0]@43..44#ROOT2024 , [alone]
            LITER 42:Root[0000, 0]@45..61#ROOT2024 Str suffixedsuffix
            PUNCT 42:Root[0000, 0]@61..62#ROOT2024 , [alone]
            LITER 42:Root[0000, 0]@63..73#ROOT2024 StrRaw(2) raw
            PUNCT 42:Root[0000, 0]@73..74#ROOT2024 , [alone]
            LITER 42:Root[0000, 0]@75..78#ROOT2024 Char a
            PUNCT 42:Root[0000, 0]@78..79#ROOT2024 , [alone]
            LITER 42:Root[0000, 0]@80..84#ROOT2024 Byte b
            PUNCT 42:Root[0000, 0]@84..85#ROOT2024 , [alone]
            LITER 42:Root[0000, 0]@86..93#ROOT2024 CStr null
        "#]],
    );
}

#[test]
fn test_fn_like_macro_negative_literals() {
    assert_expand(
        "fn_like_clone_tokens",
        r###"-1u16, - 2_u32, -3.14f32, - 2.7"###,
        expect![[r#"
            PUNCT 1 - [alone]
            LITER 1 Integer 1u16
            PUNCT 1 , [alone]
            PUNCT 1 - [alone]
            LITER 1 Integer 2_u32
            PUNCT 1 , [alone]
            PUNCT 1 - [alone]
            LITER 1 Float 3.14f32
            PUNCT 1 , [alone]
            PUNCT 1 - [alone]
            LITER 1 Float 2.7


            PUNCT 1 - [alone]
            LITER 1 Integer 1u16
            PUNCT 1 , [alone]
            PUNCT 1 - [alone]
            LITER 1 Integer 2_u32
            PUNCT 1 , [alone]
            PUNCT 1 - [alone]
            LITER 1 Float 3.14f32
            PUNCT 1 , [alone]
            PUNCT 1 - [alone]
            LITER 1 Float 2.7
        "#]],
        expect![[r#"
            PUNCT 42:Root[0000, 0]@0..1#ROOT2024 - [alone]
            LITER 42:Root[0000, 0]@1..5#ROOT2024 Integer 1u16
            PUNCT 42:Root[0000, 0]@5..6#ROOT2024 , [alone]
            PUNCT 42:Root[0000, 0]@7..8#ROOT2024 - [alone]
            LITER 42:Root[0000, 0]@9..14#ROOT2024 Integer 2_u32
            PUNCT 42:Root[0000, 0]@14..15#ROOT2024 , [alone]
            PUNCT 42:Root[0000, 0]@16..17#ROOT2024 - [alone]
            LITER 42:Root[0000, 0]@17..24#ROOT2024 Float 3.14f32
            PUNCT 42:Root[0000, 0]@24..25#ROOT2024 , [alone]
            PUNCT 42:Root[0000, 0]@26..27#ROOT2024 - [alone]
            LITER 42:Root[0000, 0]@28..31#ROOT2024 Float 2.7


            PUNCT 42:Root[0000, 0]@0..1#ROOT2024 - [alone]
            LITER 42:Root[0000, 0]@1..5#ROOT2024 Integer 1u16
            PUNCT 42:Root[0000, 0]@5..6#ROOT2024 , [alone]
            PUNCT 42:Root[0000, 0]@7..8#ROOT2024 - [alone]
            LITER 42:Root[0000, 0]@9..14#ROOT2024 Integer 2_u32
            PUNCT 42:Root[0000, 0]@14..15#ROOT2024 , [alone]
            PUNCT 42:Root[0000, 0]@16..17#ROOT2024 - [alone]
            LITER 42:Root[0000, 0]@17..24#ROOT2024 Float 3.14f32
            PUNCT 42:Root[0000, 0]@24..25#ROOT2024 , [alone]
            PUNCT 42:Root[0000, 0]@26..27#ROOT2024 - [alone]
            LITER 42:Root[0000, 0]@28..31#ROOT2024 Float 2.7
        "#]],
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
            IDENT 1 mod
            IDENT 1 m
            GROUP {} 1 1 1


            IDENT 1 some
            IDENT 1 arguments


            IDENT 1 compile_error
            PUNCT 1 ! [joint]
            GROUP () 1 1 1
              LITER 1 Str #[attr_error(some arguments)] mod m {}
            PUNCT 1 ; [alone]
        "#]],
        expect![[r#"
            IDENT 42:Root[0000, 0]@0..3#ROOT2024 mod
            IDENT 42:Root[0000, 0]@4..5#ROOT2024 m
            GROUP {} 42:Root[0000, 0]@6..7#ROOT2024 42:Root[0000, 0]@7..8#ROOT2024 42:Root[0000, 0]@6..8#ROOT2024


            IDENT 42:Root[0000, 0]@0..4#ROOT2024 some
            IDENT 42:Root[0000, 0]@5..14#ROOT2024 arguments


            IDENT 42:Root[0000, 0]@0..13#ROOT2024 compile_error
            PUNCT 42:Root[0000, 0]@13..14#ROOT2024 ! [joint]
            GROUP () 42:Root[0000, 0]@14..15#ROOT2024 42:Root[0000, 0]@55..56#ROOT2024 42:Root[0000, 0]@14..56#ROOT2024
              LITER 42:Root[0000, 0]@15..55#ROOT2024 Str #[attr_error(some arguments)] mod m {}
            PUNCT 42:Root[0000, 0]@56..57#ROOT2024 ; [alone]
        "#]],
    );
}

#[test]
#[should_panic = "called `Result::unwrap()` on an `Err` value: \"Mismatched token groups\""]
fn test_broken_input_unclosed_delim() {
    assert_expand("fn_like_clone_tokens", r###"{"###, expect![[]], expect![[]]);
}

#[test]
#[should_panic = "called `Result::unwrap()` on an `Err` value: \"Unexpected '}'\""]
fn test_broken_input_unopened_delim() {
    assert_expand("fn_like_clone_tokens", r###"}"###, expect![[]], expect![[]]);
}

#[test]
#[should_panic = "called `Result::unwrap()` on an `Err` value: \"Expected '}'\""]
fn test_broken_input_mismatched_delim() {
    assert_expand("fn_like_clone_tokens", r###"(}"###, expect![[]], expect![[]]);
}

#[test]
#[should_panic = "called `Result::unwrap()` on an `Err` value: \"Invalid identifier: `🪟`\""]
fn test_broken_input_unknowm_token() {
    assert_expand("fn_like_clone_tokens", r###"🪟"###, expect![[]], expect![[]]);
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
        DeriveReemit [CustomDerive]
        DeriveEmpty [CustomDerive]
        DerivePanic [CustomDerive]
        DeriveError [CustomDerive]"#]]
    .assert_eq(&res);
}
