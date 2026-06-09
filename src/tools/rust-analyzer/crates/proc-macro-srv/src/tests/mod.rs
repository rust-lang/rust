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
            IDENT 42:Root[0000, 0]@0..6#0 struct
            IDENT 42:Root[0000, 0]@7..8#0 S
            GROUP {} 42:Root[0000, 0]@9..10#0 42:Root[0000, 0]@46..47#0 42:Root[0000, 0]@9..47#0
              IDENT 42:Root[0000, 0]@11..16#0 field
              PUNCT 42:Root[0000, 0]@16..17#0 : [alone]
              PUNCT 42:Root[0000, 0]@18..19#0 & [joint]
              PUNCT 42:Root[0000, 0]@22..23#0 ' [joint]
              IDENT 42:Root[0000, 0]@22..24#0 r#lt
              IDENT 42:Root[0000, 0]@25..27#0 fn
              GROUP () 42:Root[0000, 0]@27..28#0 42:Root[0000, 0]@31..32#0 42:Root[0000, 0]@27..32#0
                IDENT 42:Root[0000, 0]@28..31#0 u32
              PUNCT 42:Root[0000, 0]@33..34#0 - [joint]
              PUNCT 42:Root[0000, 0]@34..35#0 > [alone]
              PUNCT 42:Root[0000, 0]@36..37#0 & [joint]
              PUNCT 42:Root[0000, 0]@38..39#0 ' [joint]
              IDENT 42:Root[0000, 0]@38..39#0 a
              IDENT 42:Root[0000, 0]@42..45#0 r#u32
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
            PUNCT 42:Root[0000, 0]@1..2#0 # [joint]
            GROUP [] 42:Root[0000, 0]@2..3#0 42:Root[0000, 0]@52..53#0 42:Root[0000, 0]@2..53#0
              IDENT 42:Root[0000, 0]@3..9#0 helper
              GROUP () 42:Root[0000, 0]@9..10#0 42:Root[0000, 0]@51..52#0 42:Root[0000, 0]@9..52#0
                IDENT 42:Root[0000, 0]@10..18#0 build_fn
                GROUP () 42:Root[0000, 0]@18..19#0 42:Root[0000, 0]@50..51#0 42:Root[0000, 0]@18..51#0
                  IDENT 42:Root[0000, 0]@19..26#0 private
                  PUNCT 42:Root[0000, 0]@26..27#0 , [alone]
                  IDENT 42:Root[0000, 0]@28..32#0 name
                  PUNCT 42:Root[0000, 0]@33..34#0 = [alone]
                  LITER 42:Root[0000, 0]@35..50#0 Str partial_build
            IDENT 42:Root[0000, 0]@54..57#0 pub
            IDENT 42:Root[0000, 0]@58..64#0 struct
            IDENT 42:Root[0000, 0]@65..68#0 Foo
            GROUP {} 42:Root[0000, 0]@69..70#0 42:Root[0000, 0]@190..191#0 42:Root[0000, 0]@69..191#0
              PUNCT 42:Root[0000, 0]@0..0#0 # [alone]
              GROUP [] 42:Root[0000, 0]@0..0#0 42:Root[0000, 0]@0..0#0 42:Root[0000, 0]@0..0#0
                IDENT 42:Root[0000, 0]@0..0#0 doc
                PUNCT 42:Root[0000, 0]@0..0#0 = [alone]
                LITER 42:Root[0000, 0]@75..130#0 Str / The domain where this federated instance is running
              PUNCT 42:Root[0000, 0]@135..136#0 # [joint]
              GROUP [] 42:Root[0000, 0]@136..137#0 42:Root[0000, 0]@157..158#0 42:Root[0000, 0]@136..158#0
                IDENT 42:Root[0000, 0]@137..143#0 helper
                GROUP () 42:Root[0000, 0]@143..144#0 42:Root[0000, 0]@156..157#0 42:Root[0000, 0]@143..157#0
                  IDENT 42:Root[0000, 0]@144..150#0 setter
                  GROUP () 42:Root[0000, 0]@150..151#0 42:Root[0000, 0]@155..156#0 42:Root[0000, 0]@150..156#0
                    IDENT 42:Root[0000, 0]@151..155#0 into
              IDENT 42:Root[0000, 0]@163..166#0 pub
              GROUP () 42:Root[0000, 0]@166..167#0 42:Root[0000, 0]@172..173#0 42:Root[0000, 0]@166..173#0
                IDENT 42:Root[0000, 0]@167..172#0 crate
              IDENT 42:Root[0000, 0]@174..180#0 domain
              PUNCT 42:Root[0000, 0]@180..181#0 : [alone]
              IDENT 42:Root[0000, 0]@182..188#0 String
              PUNCT 42:Root[0000, 0]@188..189#0 , [alone]


            PUNCT 42:Root[0000, 0]@1..2#0 # [joint]
            GROUP [] 42:Root[0000, 0]@2..3#0 42:Root[0000, 0]@52..53#0 42:Root[0000, 0]@2..53#0
              IDENT 42:Root[0000, 0]@3..9#0 helper
              GROUP () 42:Root[0000, 0]@9..10#0 42:Root[0000, 0]@51..52#0 42:Root[0000, 0]@9..52#0
                IDENT 42:Root[0000, 0]@10..18#0 build_fn
                GROUP () 42:Root[0000, 0]@18..19#0 42:Root[0000, 0]@50..51#0 42:Root[0000, 0]@18..51#0
                  IDENT 42:Root[0000, 0]@19..26#0 private
                  PUNCT 42:Root[0000, 0]@26..27#0 , [alone]
                  IDENT 42:Root[0000, 0]@28..32#0 name
                  PUNCT 42:Root[0000, 0]@33..34#0 = [alone]
                  LITER 42:Root[0000, 0]@35..50#0 Str partial_build
            IDENT 42:Root[0000, 0]@54..57#0 pub
            IDENT 42:Root[0000, 0]@58..64#0 struct
            IDENT 42:Root[0000, 0]@65..68#0 Foo
            GROUP {} 42:Root[0000, 0]@69..70#0 42:Root[0000, 0]@190..191#0 42:Root[0000, 0]@69..191#0
              PUNCT 42:Root[0000, 0]@0..0#0 # [alone]
              GROUP [] 42:Root[0000, 0]@0..0#0 42:Root[0000, 0]@0..0#0 42:Root[0000, 0]@0..0#0
                IDENT 42:Root[0000, 0]@0..0#0 doc
                PUNCT 42:Root[0000, 0]@0..0#0 = [alone]
                LITER 42:Root[0000, 0]@75..130#0 Str / The domain where this federated instance is running
              PUNCT 42:Root[0000, 0]@135..136#0 # [joint]
              GROUP [] 42:Root[0000, 0]@136..137#0 42:Root[0000, 0]@157..158#0 42:Root[0000, 0]@136..158#0
                IDENT 42:Root[0000, 0]@137..143#0 helper
                GROUP () 42:Root[0000, 0]@143..144#0 42:Root[0000, 0]@156..157#0 42:Root[0000, 0]@143..157#0
                  IDENT 42:Root[0000, 0]@144..150#0 setter
                  GROUP () 42:Root[0000, 0]@150..151#0 42:Root[0000, 0]@155..156#0 42:Root[0000, 0]@150..156#0
                    IDENT 42:Root[0000, 0]@151..155#0 into
              IDENT 42:Root[0000, 0]@163..166#0 pub
              GROUP () 42:Root[0000, 0]@166..167#0 42:Root[0000, 0]@172..173#0 42:Root[0000, 0]@166..173#0
                IDENT 42:Root[0000, 0]@167..172#0 crate
              IDENT 42:Root[0000, 0]@174..180#0 domain
              PUNCT 42:Root[0000, 0]@180..181#0 : [alone]
              IDENT 42:Root[0000, 0]@182..188#0 String
              PUNCT 42:Root[0000, 0]@188..189#0 , [alone]
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
            IDENT 42:Root[0000, 0]@0..6#0 struct
            IDENT 42:Root[0000, 0]@7..8#0 S
            GROUP {} 42:Root[0000, 0]@9..10#0 42:Root[0000, 0]@22..23#0 42:Root[0000, 0]@9..23#0
              IDENT 42:Root[0000, 0]@11..16#0 field
              PUNCT 42:Root[0000, 0]@16..17#0 : [alone]
              IDENT 42:Root[0000, 0]@18..21#0 u32


            IDENT 42:Root[0000, 0]@0..13#0 compile_error
            PUNCT 42:Root[0000, 0]@13..14#0 ! [joint]
            GROUP () 42:Root[0000, 0]@14..15#0 42:Root[0000, 0]@62..63#0 42:Root[0000, 0]@14..63#0
              LITER 42:Root[0000, 0]@15..62#0 Str #[derive(DeriveError)] struct S {field : u32}
            PUNCT 42:Root[0000, 0]@63..64#0 ; [alone]
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
            IDENT 42:Root[0000, 0]@0..5#0 ident
            PUNCT 42:Root[0000, 0]@5..6#0 , [alone]
            LITER 42:Root[0000, 0]@7..8#0 Integer 0
            PUNCT 42:Root[0000, 0]@8..9#0 , [alone]
            LITER 42:Root[0000, 0]@10..11#0 Integer 1
            PUNCT 42:Root[0000, 0]@11..12#0 , [alone]
            GROUP [] 42:Root[0000, 0]@13..14#0 42:Root[0000, 0]@14..15#0 42:Root[0000, 0]@13..15#0


            IDENT 42:Root[0000, 0]@0..5#0 ident
            PUNCT 42:Root[0000, 0]@5..6#0 , [alone]
            LITER 42:Root[0000, 0]@7..8#0 Integer 0
            PUNCT 42:Root[0000, 0]@8..9#0 , [alone]
            LITER 42:Root[0000, 0]@10..11#0 Integer 1
            PUNCT 42:Root[0000, 0]@11..12#0 , [alone]
            GROUP [] 42:Root[0000, 0]@13..14#0 42:Root[0000, 0]@14..15#0 42:Root[0000, 0]@13..15#0
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
            IDENT 42:Root[0000, 0]@0..5#0 ident
            PUNCT 42:Root[0000, 0]@5..6#0 , [alone]
            GROUP [] 42:Root[0000, 0]@7..8#0 42:Root[0000, 0]@22..23#0 42:Root[0000, 0]@7..23#0
              IDENT 42:Root[0000, 0]@8..14#0 ident2
              PUNCT 42:Root[0000, 0]@14..15#0 , [alone]
              IDENT 42:Root[0000, 0]@16..22#0 ident3


            IDENT 42:Root[0000, 0]@0..5#0 ident
            PUNCT 42:Root[0000, 0]@5..6#0 , [alone]
            GROUP [] 42:Root[0000, 0]@7..23#0 42:Root[0000, 0]@7..23#0 42:Root[0000, 0]@7..23#0
              IDENT 42:Root[0000, 0]@8..14#0 ident2
              PUNCT 42:Root[0000, 0]@14..15#0 , [alone]
              IDENT 42:Root[0000, 0]@16..22#0 ident3
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
            IDENT 42:Root[0000, 0]@2..7#0 r#async


            IDENT 42:Root[0000, 0]@2..7#0 r#async
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
            IDENT 42:Root[0000, 0]@0..3#0 foo
            IDENT 42:Root[0000, 0]@8..11#0 bar


            IDENT 42:Root[0000, 0]@0..11#0 r#joined
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
            IDENT 42:Root[0000, 0]@0..12#0 set_def_site
            IDENT 42:Root[0000, 0]@13..33#0 resolved_at_def_site
            IDENT 42:Root[0000, 0]@34..44#0 start_span


            IDENT 41:Root[0000, 0]@0..150#0 set_def_site
            IDENT 42:Root[0000, 0]@13..33#0 resolved_at_def_site
            IDENT 42:Root[0000, 0]@34..34#0 start_span
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


            LITER 42:Root[0000, 0]@0..100#0 ByteStr byte_string
            LITER 42:Root[0000, 0]@0..100#0 Char c
            LITER 42:Root[0000, 0]@0..100#0 Str string
            LITER 42:Root[0000, 0]@0..100#0 Str -string
            LITER 42:Root[0000, 0]@0..100#0 CStr cstring
            LITER 42:Root[0000, 0]@0..100#0 Float 3.14f64
            LITER 42:Root[0000, 0]@0..100#0 Float -3.14f64
            LITER 42:Root[0000, 0]@0..100#0 Float 3.14
            LITER 42:Root[0000, 0]@0..100#0 Float -3.14
            LITER 42:Root[0000, 0]@0..100#0 Integer 123i64
            LITER 42:Root[0000, 0]@0..100#0 Integer -123i64
            LITER 42:Root[0000, 0]@0..100#0 Integer 123
            LITER 42:Root[0000, 0]@0..100#0 Integer -123
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


            IDENT 42:Root[0000, 0]@0..100#0 standard
            IDENT 42:Root[0000, 0]@0..100#0 r#raw
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
            LITER 42:Root[0000, 0]@0..4#0 Integer 1u16
            PUNCT 42:Root[0000, 0]@4..5#0 , [alone]
            LITER 42:Root[0000, 0]@6..11#0 Integer 2_u32
            PUNCT 42:Root[0000, 0]@11..12#0 , [alone]
            PUNCT 42:Root[0000, 0]@13..14#0 - [alone]
            LITER 42:Root[0000, 0]@14..18#0 Integer 4i64
            PUNCT 42:Root[0000, 0]@18..19#0 , [alone]
            LITER 42:Root[0000, 0]@20..27#0 Float 3.14f32
            PUNCT 42:Root[0000, 0]@27..28#0 , [alone]
            LITER 42:Root[0000, 0]@29..43#0 Str hello bridge
            PUNCT 42:Root[0000, 0]@43..44#0 , [alone]
            LITER 42:Root[0000, 0]@45..61#0 Str suffixedsuffix
            PUNCT 42:Root[0000, 0]@61..62#0 , [alone]
            LITER 42:Root[0000, 0]@63..73#0 StrRaw(2) raw
            PUNCT 42:Root[0000, 0]@73..74#0 , [alone]
            LITER 42:Root[0000, 0]@75..78#0 Char a
            PUNCT 42:Root[0000, 0]@78..79#0 , [alone]
            LITER 42:Root[0000, 0]@80..84#0 Byte b
            PUNCT 42:Root[0000, 0]@84..85#0 , [alone]
            LITER 42:Root[0000, 0]@86..93#0 CStr null


            LITER 42:Root[0000, 0]@0..4#0 Integer 1u16
            PUNCT 42:Root[0000, 0]@4..5#0 , [alone]
            LITER 42:Root[0000, 0]@6..11#0 Integer 2_u32
            PUNCT 42:Root[0000, 0]@11..12#0 , [alone]
            PUNCT 42:Root[0000, 0]@13..14#0 - [alone]
            LITER 42:Root[0000, 0]@14..18#0 Integer 4i64
            PUNCT 42:Root[0000, 0]@18..19#0 , [alone]
            LITER 42:Root[0000, 0]@20..27#0 Float 3.14f32
            PUNCT 42:Root[0000, 0]@27..28#0 , [alone]
            LITER 42:Root[0000, 0]@29..43#0 Str hello bridge
            PUNCT 42:Root[0000, 0]@43..44#0 , [alone]
            LITER 42:Root[0000, 0]@45..61#0 Str suffixedsuffix
            PUNCT 42:Root[0000, 0]@61..62#0 , [alone]
            LITER 42:Root[0000, 0]@63..73#0 StrRaw(2) raw
            PUNCT 42:Root[0000, 0]@73..74#0 , [alone]
            LITER 42:Root[0000, 0]@75..78#0 Char a
            PUNCT 42:Root[0000, 0]@78..79#0 , [alone]
            LITER 42:Root[0000, 0]@80..84#0 Byte b
            PUNCT 42:Root[0000, 0]@84..85#0 , [alone]
            LITER 42:Root[0000, 0]@86..93#0 CStr null
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
            PUNCT 42:Root[0000, 0]@0..1#0 - [alone]
            LITER 42:Root[0000, 0]@1..5#0 Integer 1u16
            PUNCT 42:Root[0000, 0]@5..6#0 , [alone]
            PUNCT 42:Root[0000, 0]@7..8#0 - [alone]
            LITER 42:Root[0000, 0]@9..14#0 Integer 2_u32
            PUNCT 42:Root[0000, 0]@14..15#0 , [alone]
            PUNCT 42:Root[0000, 0]@16..17#0 - [alone]
            LITER 42:Root[0000, 0]@17..24#0 Float 3.14f32
            PUNCT 42:Root[0000, 0]@24..25#0 , [alone]
            PUNCT 42:Root[0000, 0]@26..27#0 - [alone]
            LITER 42:Root[0000, 0]@28..31#0 Float 2.7


            PUNCT 42:Root[0000, 0]@0..1#0 - [alone]
            LITER 42:Root[0000, 0]@1..5#0 Integer 1u16
            PUNCT 42:Root[0000, 0]@5..6#0 , [alone]
            PUNCT 42:Root[0000, 0]@7..8#0 - [alone]
            LITER 42:Root[0000, 0]@9..14#0 Integer 2_u32
            PUNCT 42:Root[0000, 0]@14..15#0 , [alone]
            PUNCT 42:Root[0000, 0]@16..17#0 - [alone]
            LITER 42:Root[0000, 0]@17..24#0 Float 3.14f32
            PUNCT 42:Root[0000, 0]@24..25#0 , [alone]
            PUNCT 42:Root[0000, 0]@26..27#0 - [alone]
            LITER 42:Root[0000, 0]@28..31#0 Float 2.7
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
            IDENT 42:Root[0000, 0]@0..3#0 mod
            IDENT 42:Root[0000, 0]@4..5#0 m
            GROUP {} 42:Root[0000, 0]@6..7#0 42:Root[0000, 0]@7..8#0 42:Root[0000, 0]@6..8#0


            IDENT 42:Root[0000, 0]@0..4#0 some
            IDENT 42:Root[0000, 0]@5..14#0 arguments


            IDENT 42:Root[0000, 0]@0..13#0 compile_error
            PUNCT 42:Root[0000, 0]@13..14#0 ! [joint]
            GROUP () 42:Root[0000, 0]@14..15#0 42:Root[0000, 0]@55..56#0 42:Root[0000, 0]@14..56#0
              LITER 42:Root[0000, 0]@15..55#0 Str #[attr_error(some arguments)] mod m {}
            PUNCT 42:Root[0000, 0]@56..57#0 ; [alone]
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
        fn_like_span_line_column [Bang]
        attr_noop [Attr]
        attr_panic [Attr]
        attr_error [Attr]
        DeriveReemit [CustomDerive]
        DeriveEmpty [CustomDerive]
        DerivePanic [CustomDerive]
        DeriveError [CustomDerive]"#]]
    .assert_eq(&res);
}

#[test]
fn test_fn_like_span_line_column() {
    assert_expand_with_callback(
        "fn_like_span_line_column",
        // Input text with known position: "hello" starts at offset 1 (line 2, column 1 in 1-based)
        "
hello",
        expect![[r#"
            LITER 42:Root[0000, 0]@0..100#0 Integer 2
            LITER 42:Root[0000, 0]@0..100#0 Integer 1
        "#]],
    );
}
