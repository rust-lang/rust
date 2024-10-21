//! Tests specific to declarative macros, aka macros by example. This covers
//! both stable `macro_rules!` macros as well as unstable `macro` macros.
// FIXME: Move more of the nameres independent tests from
// crates\hir-def\src\macro_expansion_tests\mod.rs to this
use expect_test::expect;
use span::{Edition, EditionedFileId, ErasedFileAstId, FileId, Span, SpanAnchor, SyntaxContextId};
use stdx::format_to;
use tt::{TextRange, TextSize};

use crate::DeclarativeMacro;

#[expect(deprecated)]
fn check_(
    def_edition: Edition,
    call_edition: Edition,
    macro2: bool,
    decl: &str,
    arg: &str,
    render_debug: bool,
    expect: expect_test::Expect,
    parse: parser::TopEntryPoint,
) {
    let decl_tt = &syntax_bridge::parse_to_token_tree(
        def_edition,
        SpanAnchor {
            file_id: EditionedFileId::new(FileId::from_raw(0), def_edition),
            ast_id: ErasedFileAstId::from_raw(0),
        },
        SyntaxContextId::ROOT,
        decl,
    )
    .unwrap();
    let mac = if macro2 {
        DeclarativeMacro::parse_macro2(None, decl_tt, |_| def_edition)
    } else {
        DeclarativeMacro::parse_macro_rules(decl_tt, |_| def_edition)
    };
    let call_anchor = SpanAnchor {
        file_id: EditionedFileId::new(FileId::from_raw(1), call_edition),
        ast_id: ErasedFileAstId::from_raw(0),
    };
    let arg_tt =
        syntax_bridge::parse_to_token_tree(call_edition, call_anchor, SyntaxContextId::ROOT, arg)
            .unwrap();
    let res = mac.expand(
        &arg_tt,
        |_| (),
        Span {
            range: TextRange::up_to(TextSize::of(arg)),
            anchor: call_anchor,
            ctx: SyntaxContextId::ROOT,
        },
        def_edition,
    );
    let mut expect_res = String::new();
    if let Some(err) = res.err {
        format_to!(expect_res, "{err:#?}\n\n",);
    }
    if render_debug {
        format_to!(expect_res, "{:#?}\n\n", res.value.0);
    }
    let (node, _) = syntax_bridge::token_tree_to_syntax_node(&res.value.0, parse, def_edition);
    format_to!(
        expect_res,
        "{}",
        syntax_bridge::prettify_macro_expansion::prettify_macro_expansion(
            node.syntax_node(),
            &mut |it| it.clone()
        )
    );
    expect.assert_eq(&expect_res);
}

fn check(
    def_edition: Edition,
    call_edition: Edition,
    decl: &str,
    arg: &str,
    expect: expect_test::Expect,
) {
    check_(
        def_edition,
        call_edition,
        false,
        decl,
        arg,
        true,
        expect,
        parser::TopEntryPoint::SourceFile,
    );
}

#[test]
fn token_mapping_smoke_test() {
    check(
        Edition::CURRENT,
        Edition::CURRENT,
        r#"
( struct $ident:ident ) => {
    struct $ident {
        map: ::std::collections::HashSet<()>,
    }
};
"#,
        r#"
struct MyTraitMap2
"#,
        expect![[r#"
            SUBTREE $$ 1:0@0..20#0 1:0@0..20#0
              IDENT   struct 0:0@34..40#0
              IDENT   MyTraitMap2 1:0@8..19#0
              SUBTREE {} 0:0@48..49#0 0:0@100..101#0
                IDENT   map 0:0@58..61#0
                PUNCH   : [alone] 0:0@61..62#0
                PUNCH   : [joint] 0:0@63..64#0
                PUNCH   : [alone] 0:0@64..65#0
                IDENT   std 0:0@65..68#0
                PUNCH   : [joint] 0:0@68..69#0
                PUNCH   : [alone] 0:0@69..70#0
                IDENT   collections 0:0@70..81#0
                PUNCH   : [joint] 0:0@81..82#0
                PUNCH   : [alone] 0:0@82..83#0
                IDENT   HashSet 0:0@83..90#0
                PUNCH   < [alone] 0:0@90..91#0
                SUBTREE () 0:0@91..92#0 0:0@92..93#0
                PUNCH   > [joint] 0:0@93..94#0
                PUNCH   , [alone] 0:0@94..95#0

            struct MyTraitMap2 {
                map: ::std::collections::HashSet<()>,
            }"#]],
    );
}

#[test]
fn token_mapping_floats() {
    // Regression test for https://github.com/rust-lang/rust-analyzer/issues/12216
    // (and related issues)
    check(
        Edition::CURRENT,
        Edition::CURRENT,
        r#"
($($tt:tt)*) => {
    $($tt)*
};
"#,
        r#"
fn main() {
    1;
    1.0;
    ((1,),).0.0;
    let x = 1;
}
"#,
        expect![[r#"
            SUBTREE $$ 1:0@0..63#0 1:0@0..63#0
              IDENT   fn 1:0@1..3#0
              IDENT   main 1:0@4..8#0
              SUBTREE () 1:0@8..9#0 1:0@9..10#0
              SUBTREE {} 1:0@11..12#0 1:0@61..62#0
                LITERAL Integer 1 1:0@17..18#0
                PUNCH   ; [alone] 1:0@18..19#0
                LITERAL Float 1.0 1:0@24..27#0
                PUNCH   ; [alone] 1:0@27..28#0
                SUBTREE () 1:0@33..34#0 1:0@39..40#0
                  SUBTREE () 1:0@34..35#0 1:0@37..38#0
                    LITERAL Integer 1 1:0@35..36#0
                    PUNCH   , [alone] 1:0@36..37#0
                  PUNCH   , [alone] 1:0@38..39#0
                PUNCH   . [alone] 1:0@40..41#0
                LITERAL Float 0.0 1:0@41..44#0
                PUNCH   ; [alone] 1:0@44..45#0
                IDENT   let 1:0@50..53#0
                IDENT   x 1:0@54..55#0
                PUNCH   = [alone] 1:0@56..57#0
                LITERAL Integer 1 1:0@58..59#0
                PUNCH   ; [alone] 1:0@59..60#0

            fn main(){
                1;
                1.0;
                ((1,),).0.0;
                let x = 1;
            }"#]],
    );
}

#[test]
fn expr_2021() {
    check(
        Edition::Edition2024,
        Edition::Edition2024,
        r#"
($($e:expr),* $(,)?) => {
    $($e);* ;
};
"#,
        r#"
    _,
    const { 1 },
"#,
        expect![[r#"
            SUBTREE $$ 1:0@0..25#0 1:0@0..25#0
              IDENT   _ 1:0@5..6#0
              PUNCH   ; [joint] 0:0@36..37#0
              SUBTREE () 0:0@34..35#0 0:0@34..35#0
                IDENT   const 1:0@12..17#0
                SUBTREE {} 1:0@18..19#0 1:0@22..23#0
                  LITERAL Integer 1 1:0@20..21#0
              PUNCH   ; [alone] 0:0@39..40#0

            _;
            (const  {
                1
            });"#]],
    );
    check(
        Edition::Edition2021,
        Edition::Edition2024,
        r#"
($($e:expr),* $(,)?) => {
    $($e);* ;
};
"#,
        r#"
    _,
"#,
        expect![[r#"
            ExpandError {
                inner: (
                    1:0@5..6#0,
                    NoMatchingRule,
                ),
            }

            SUBTREE $$ 1:0@0..8#0 1:0@0..8#0
              PUNCH   ; [alone] 0:0@39..40#0

            ;"#]],
    );
    check(
        Edition::Edition2021,
        Edition::Edition2024,
        r#"
($($e:expr),* $(,)?) => {
    $($e);* ;
};
"#,
        r#"
    const { 1 },
"#,
        expect![[r#"
            ExpandError {
                inner: (
                    1:0@5..10#0,
                    NoMatchingRule,
                ),
            }

            SUBTREE $$ 1:0@0..18#0 1:0@0..18#0
              PUNCH   ; [alone] 0:0@39..40#0

            ;"#]],
    );
    check(
        Edition::Edition2024,
        Edition::Edition2024,
        r#"
($($e:expr_2021),* $(,)?) => {
    $($e);* ;
};
"#,
        r#"
    4,
    "literal",
    funcall(),
    future.await,
    break 'foo bar,
"#,
        expect![[r#"
            SUBTREE $$ 1:0@0..76#0 1:0@0..76#0
              LITERAL Integer 4 1:0@5..6#0
              PUNCH   ; [joint] 0:0@41..42#0
              LITERAL Str literal 1:0@12..21#0
              PUNCH   ; [joint] 0:0@41..42#0
              SUBTREE () 0:0@39..40#0 0:0@39..40#0
                IDENT   funcall 1:0@27..34#0
                SUBTREE () 1:0@34..35#0 1:0@35..36#0
              PUNCH   ; [joint] 0:0@41..42#0
              SUBTREE () 0:0@39..40#0 0:0@39..40#0
                IDENT   future 1:0@42..48#0
                PUNCH   . [alone] 1:0@48..49#0
                IDENT   await 1:0@49..54#0
              PUNCH   ; [joint] 0:0@41..42#0
              SUBTREE () 0:0@39..40#0 0:0@39..40#0
                IDENT   break 1:0@60..65#0
                PUNCH   ' [joint] 1:0@66..67#0
                IDENT   foo 1:0@67..70#0
                IDENT   bar 1:0@71..74#0
              PUNCH   ; [alone] 0:0@44..45#0

            4;
            "literal";
            (funcall());
            (future.await);
            (break 'foo bar);"#]],
    );
    check(
        Edition::Edition2024,
        Edition::Edition2024,
        r#"
($($e:expr_2021),* $(,)?) => {
    $($e);* ;
};
"#,
        r#"
    _,
"#,
        expect![[r#"
            ExpandError {
                inner: (
                    1:0@5..6#0,
                    NoMatchingRule,
                ),
            }

            SUBTREE $$ 1:0@0..8#0 1:0@0..8#0
              PUNCH   ; [alone] 0:0@44..45#0

            ;"#]],
    );
}
