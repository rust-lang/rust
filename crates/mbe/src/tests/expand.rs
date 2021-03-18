use ::parser::FragmentKind;
use syntax::{
    SyntaxKind::{ERROR, IDENT},
    T,
};
use test_utils::assert_eq_text;

use super::*;

// Good first issue (although a slightly challenging one):
//
// * Pick a random test from here
//   https://github.com/intellij-rust/intellij-rust/blob/c4e9feee4ad46e7953b1948c112533360b6087bb/src/test/kotlin/org/rust/lang/core/macros/RsMacroExpansionTest.kt
// * Port the test to rust and add it to this module
// * Make it pass :-)

#[test]
fn test_token_id_shift() {
    let expansion = parse_macro(
        r#"
macro_rules! foobar {
    ($e:ident) => { foo bar $e }
}
"#,
    )
    .expand_tt("foobar!(baz);");

    fn get_id(t: &tt::TokenTree) -> Option<u32> {
        if let tt::TokenTree::Leaf(tt::Leaf::Ident(ident)) = t {
            return Some(ident.id.0);
        }
        None
    }

    assert_eq!(expansion.token_trees.len(), 3);
    // {($e:ident) => { foo bar $e }}
    // 012345      67 8 9   T   12
    assert_eq!(get_id(&expansion.token_trees[0]), Some(9));
    assert_eq!(get_id(&expansion.token_trees[1]), Some(10));

    // The input args of macro call include parentheses:
    // (baz)
    // So baz should be 12+1+1
    assert_eq!(get_id(&expansion.token_trees[2]), Some(14));
}

#[test]
fn test_token_map() {
    let expanded = parse_macro(
        r#"
macro_rules! foobar {
    ($e:ident) => { fn $e() {} }
}
"#,
    )
    .expand_tt("foobar!(baz);");

    let (node, token_map) = token_tree_to_syntax_node(&expanded, FragmentKind::Items).unwrap();
    let content = node.syntax_node().to_string();

    let get_text = |id, kind| -> String {
        content[token_map.range_by_token(id).unwrap().by_kind(kind).unwrap()].to_string()
    };

    assert_eq!(expanded.token_trees.len(), 4);
    // {($e:ident) => { fn $e() {} }}
    // 012345      67 8 9  T12  3

    assert_eq!(get_text(tt::TokenId(9), IDENT), "fn");
    assert_eq!(get_text(tt::TokenId(12), T!['(']), "(");
    assert_eq!(get_text(tt::TokenId(13), T!['{']), "{");
}

#[test]
fn test_convert_tt() {
    parse_macro(r#"
macro_rules! impl_froms {
    ($e:ident: $($v:ident),*) => {
        $(
            impl From<$v> for $e {
                fn from(it: $v) -> $e {
                    $e::$v(it)
                }
            }
        )*
    }
}
"#)
        .assert_expand_tt(
            "impl_froms!(TokenTree: Leaf, Subtree);",
            "impl From <Leaf > for TokenTree {fn from (it : Leaf) -> TokenTree {TokenTree ::Leaf (it)}} \
             impl From <Subtree > for TokenTree {fn from (it : Subtree) -> TokenTree {TokenTree ::Subtree (it)}}"
        );
}

#[test]
fn test_convert_tt2() {
    parse_macro(
        r#"
macro_rules! impl_froms {
    ($e:ident: $($v:ident),*) => {
        $(
            impl From<$v> for $e {
                fn from(it: $v) -> $e {
                    $e::$v(it)
                }
            }
        )*
    }
}
"#,
    )
    .assert_expand(
        "impl_froms!(TokenTree: Leaf, Subtree);",
        r#"
SUBTREE $
  IDENT   impl 20
  IDENT   From 21
  PUNCH   < [joint] 22
  IDENT   Leaf 53
  PUNCH   > [alone] 25
  IDENT   for 26
  IDENT   TokenTree 51
  SUBTREE {} 29
    IDENT   fn 30
    IDENT   from 31
    SUBTREE () 32
      IDENT   it 33
      PUNCH   : [alone] 34
      IDENT   Leaf 53
    PUNCH   - [joint] 37
    PUNCH   > [alone] 38
    IDENT   TokenTree 51
    SUBTREE {} 41
      IDENT   TokenTree 51
      PUNCH   : [joint] 44
      PUNCH   : [joint] 45
      IDENT   Leaf 53
      SUBTREE () 48
        IDENT   it 49
  IDENT   impl 20
  IDENT   From 21
  PUNCH   < [joint] 22
  IDENT   Subtree 55
  PUNCH   > [alone] 25
  IDENT   for 26
  IDENT   TokenTree 51
  SUBTREE {} 29
    IDENT   fn 30
    IDENT   from 31
    SUBTREE () 32
      IDENT   it 33
      PUNCH   : [alone] 34
      IDENT   Subtree 55
    PUNCH   - [joint] 37
    PUNCH   > [alone] 38
    IDENT   TokenTree 51
    SUBTREE {} 41
      IDENT   TokenTree 51
      PUNCH   : [joint] 44
      PUNCH   : [joint] 45
      IDENT   Subtree 55
      SUBTREE () 48
        IDENT   it 49
"#,
    );
}

#[test]
fn test_lifetime_split() {
    parse_macro(
        r#"
macro_rules! foo {
    ($($t:tt)*) => { $($t)*}
}
"#,
    )
    .assert_expand(
        r#"foo!(static bar: &'static str = "hello";);"#,
        r#"
SUBTREE $
  IDENT   static 17
  IDENT   bar 18
  PUNCH   : [alone] 19
  PUNCH   & [alone] 20
  PUNCH   ' [joint] 21
  IDENT   static 22
  IDENT   str 23
  PUNCH   = [alone] 24
  LITERAL "hello" 25
  PUNCH   ; [joint] 26
"#,
    );
}

#[test]
fn test_expr_order() {
    let expanded = parse_macro(
        r#"
        macro_rules! foo {
            ($ i:expr) => {
                 fn bar() { $ i * 2; }
            }
        }
"#,
    )
    .expand_items("foo! { 1 + 1}");

    let dump = format!("{:#?}", expanded);
    assert_eq_text!(
        r#"MACRO_ITEMS@0..15
  FN@0..15
    FN_KW@0..2 "fn"
    NAME@2..5
      IDENT@2..5 "bar"
    PARAM_LIST@5..7
      L_PAREN@5..6 "("
      R_PAREN@6..7 ")"
    BLOCK_EXPR@7..15
      L_CURLY@7..8 "{"
      EXPR_STMT@8..14
        BIN_EXPR@8..13
          BIN_EXPR@8..11
            LITERAL@8..9
              INT_NUMBER@8..9 "1"
            PLUS@9..10 "+"
            LITERAL@10..11
              INT_NUMBER@10..11 "1"
          STAR@11..12 "*"
          LITERAL@12..13
            INT_NUMBER@12..13 "2"
        SEMICOLON@13..14 ";"
      R_CURLY@14..15 "}""#,
        dump.trim()
    );
}

#[test]
fn test_fail_match_pattern_by_first_token() {
    parse_macro(
        r#"
        macro_rules! foo {
            ($ i:ident) => (
                mod $ i {}
            );
            (= $ i:ident) => (
                fn $ i() {}
            );
            (+ $ i:ident) => (
                struct $ i;
            )
        }
"#,
    )
    .assert_expand_items("foo! { foo }", "mod foo {}")
    .assert_expand_items("foo! { = bar }", "fn bar () {}")
    .assert_expand_items("foo! { + Baz }", "struct Baz ;");
}

#[test]
fn test_fail_match_pattern_by_last_token() {
    parse_macro(
        r#"
        macro_rules! foo {
            ($ i:ident) => (
                mod $ i {}
            );
            ($ i:ident =) => (
                fn $ i() {}
            );
            ($ i:ident +) => (
                struct $ i;
            )
        }
"#,
    )
    .assert_expand_items("foo! { foo }", "mod foo {}")
    .assert_expand_items("foo! { bar = }", "fn bar () {}")
    .assert_expand_items("foo! { Baz + }", "struct Baz ;");
}

#[test]
fn test_fail_match_pattern_by_word_token() {
    parse_macro(
        r#"
        macro_rules! foo {
            ($ i:ident) => (
                mod $ i {}
            );
            (spam $ i:ident) => (
                fn $ i() {}
            );
            (eggs $ i:ident) => (
                struct $ i;
            )
        }
"#,
    )
    .assert_expand_items("foo! { foo }", "mod foo {}")
    .assert_expand_items("foo! { spam bar }", "fn bar () {}")
    .assert_expand_items("foo! { eggs Baz }", "struct Baz ;");
}

#[test]
fn test_match_group_pattern_by_separator_token() {
    parse_macro(
        r#"
        macro_rules! foo {
            ($ ($ i:ident),*) => ($ (
                mod $ i {}
            )*);
            ($ ($ i:ident)#*) => ($ (
                fn $ i() {}
            )*);
            ($ i:ident ,# $ j:ident) => (
                struct $ i;
                struct $ j;
            )
        }
"#,
    )
    .assert_expand_items("foo! { foo, bar }", "mod foo {} mod bar {}")
    .assert_expand_items("foo! { foo# bar }", "fn foo () {} fn bar () {}")
    .assert_expand_items("foo! { Foo,# Bar }", "struct Foo ; struct Bar ;");
}

#[test]
fn test_match_group_pattern_with_multiple_defs() {
    parse_macro(
        r#"
        macro_rules! foo {
            ($ ($ i:ident),*) => ( struct Bar { $ (
                fn $ i {}
            )*} );
        }
"#,
    )
    .assert_expand_items("foo! { foo, bar }", "struct Bar {fn foo {} fn bar {}}");
}

#[test]
fn test_match_group_pattern_with_multiple_statement() {
    parse_macro(
        r#"
        macro_rules! foo {
            ($ ($ i:ident),*) => ( fn baz { $ (
                $ i ();
            )*} );
        }
"#,
    )
    .assert_expand_items("foo! { foo, bar }", "fn baz {foo () ; bar () ;}");
}

#[test]
fn test_match_group_pattern_with_multiple_statement_without_semi() {
    parse_macro(
        r#"
        macro_rules! foo {
            ($ ($ i:ident),*) => ( fn baz { $ (
                $i()
            );*} );
        }
"#,
    )
    .assert_expand_items("foo! { foo, bar }", "fn baz {foo () ;bar ()}");
}

#[test]
fn test_match_group_empty_fixed_token() {
    parse_macro(
        r#"
        macro_rules! foo {
            ($ ($ i:ident)* #abc) => ( fn baz { $ (
                $ i ();
            )*} );
        }
"#,
    )
    .assert_expand_items("foo! {#abc}", "fn baz {}");
}

#[test]
fn test_match_group_in_subtree() {
    parse_macro(
        r#"
        macro_rules! foo {
            (fn $name:ident {$($i:ident)*} ) => ( fn $name() { $ (
                $ i ();
            )*} );
        }"#,
    )
    .assert_expand_items("foo! {fn baz {a b} }", "fn baz () {a () ; b () ;}");
}

#[test]
fn test_match_group_with_multichar_sep() {
    parse_macro(
        r#"
        macro_rules! foo {
            (fn $name:ident {$($i:literal)*} ) => ( fn $name() -> bool { $($i)&&*} );
        }"#,
    )
    .assert_expand_items("foo! (fn baz {true true} );", "fn baz () -> bool {true &&true}");
}

#[test]
fn test_match_group_with_multichar_sep2() {
    parse_macro(
        r#"
        macro_rules! foo {
            (fn $name:ident {$($i:literal)&&*} ) => ( fn $name() -> bool { $($i)&&*} );
        }"#,
    )
    .assert_expand_items("foo! (fn baz {true && true} );", "fn baz () -> bool {true &&true}");
}

#[test]
fn test_match_group_zero_match() {
    parse_macro(
        r#"
        macro_rules! foo {
            ( $($i:ident)* ) => ();
        }"#,
    )
    .assert_expand_items("foo! ();", "");
}

#[test]
fn test_match_group_in_group() {
    parse_macro(
        r#"
        macro_rules! foo {
            { $( ( $($i:ident)* ) )* } => ( $( ( $($i)* ) )* );
        }"#,
    )
    .assert_expand_items("foo! ( (a b) );", "(a b)");
}

#[test]
fn test_expand_to_item_list() {
    let tree = parse_macro(
        "
            macro_rules! structs {
                ($($i:ident),*) => {
                    $(struct $i { field: u32 } )*
                }
            }
            ",
    )
    .expand_items("structs!(Foo, Bar);");
    assert_eq!(
        format!("{:#?}", tree).trim(),
        r#"
MACRO_ITEMS@0..40
  STRUCT@0..20
    STRUCT_KW@0..6 "struct"
    NAME@6..9
      IDENT@6..9 "Foo"
    RECORD_FIELD_LIST@9..20
      L_CURLY@9..10 "{"
      RECORD_FIELD@10..19
        NAME@10..15
          IDENT@10..15 "field"
        COLON@15..16 ":"
        PATH_TYPE@16..19
          PATH@16..19
            PATH_SEGMENT@16..19
              NAME_REF@16..19
                IDENT@16..19 "u32"
      R_CURLY@19..20 "}"
  STRUCT@20..40
    STRUCT_KW@20..26 "struct"
    NAME@26..29
      IDENT@26..29 "Bar"
    RECORD_FIELD_LIST@29..40
      L_CURLY@29..30 "{"
      RECORD_FIELD@30..39
        NAME@30..35
          IDENT@30..35 "field"
        COLON@35..36 ":"
        PATH_TYPE@36..39
          PATH@36..39
            PATH_SEGMENT@36..39
              NAME_REF@36..39
                IDENT@36..39 "u32"
      R_CURLY@39..40 "}""#
            .trim()
    );
}

fn to_subtree(tt: &tt::TokenTree) -> &tt::Subtree {
    if let tt::TokenTree::Subtree(subtree) = tt {
        return &subtree;
    }
    unreachable!("It is not a subtree");
}
fn to_literal(tt: &tt::TokenTree) -> &tt::Literal {
    if let tt::TokenTree::Leaf(tt::Leaf::Literal(lit)) = tt {
        return lit;
    }
    unreachable!("It is not a literal");
}

fn to_punct(tt: &tt::TokenTree) -> &tt::Punct {
    if let tt::TokenTree::Leaf(tt::Leaf::Punct(lit)) = tt {
        return lit;
    }
    unreachable!("It is not a Punct");
}

#[test]
fn test_expand_literals_to_token_tree() {
    let expansion = parse_macro(
        r#"
            macro_rules! literals {
                ($i:ident) => {
                    {
                        let a = 'c';
                        let c = 1000;
                        let f = 12E+99_f64;
                        let s = "rust1";
                    }
                }
            }
            "#,
    )
    .expand_tt("literals!(foo);");
    let stm_tokens = &to_subtree(&expansion.token_trees[0]).token_trees;

    // [let] [a] [=] ['c'] [;]
    assert_eq!(to_literal(&stm_tokens[3]).text, "'c'");
    // [let] [c] [=] [1000] [;]
    assert_eq!(to_literal(&stm_tokens[5 + 3]).text, "1000");
    // [let] [f] [=] [12E+99_f64] [;]
    assert_eq!(to_literal(&stm_tokens[10 + 3]).text, "12E+99_f64");
    // [let] [s] [=] ["rust1"] [;]
    assert_eq!(to_literal(&stm_tokens[15 + 3]).text, "\"rust1\"");
}

#[test]
fn test_attr_to_token_tree() {
    let expansion = parse_to_token_tree_by_syntax(
        r#"
            #[derive(Copy)]
            struct Foo;
            "#,
    );

    assert_eq!(to_punct(&expansion.token_trees[0]).char, '#');
    assert_eq!(
        to_subtree(&expansion.token_trees[1]).delimiter_kind(),
        Some(tt::DelimiterKind::Bracket)
    );
}

#[test]
fn test_two_idents() {
    parse_macro(
        r#"
        macro_rules! foo {
            ($ i:ident, $ j:ident) => {
                fn foo() { let a = $ i; let b = $j; }
            }
        }
"#,
    )
    .assert_expand_items("foo! { foo, bar }", "fn foo () {let a = foo ; let b = bar ;}");
}

#[test]
fn test_tt_to_stmts() {
    let stmts = parse_macro(
        r#"
        macro_rules! foo {
            () => {
                 let a = 0;
                 a = 10 + 1;
                 a
            }
        }
"#,
    )
    .expand_statements("foo!{}");

    assert_eq!(
        format!("{:#?}", stmts).trim(),
        r#"MACRO_STMTS@0..15
  LET_STMT@0..7
    LET_KW@0..3 "let"
    IDENT_PAT@3..4
      NAME@3..4
        IDENT@3..4 "a"
    EQ@4..5 "="
    LITERAL@5..6
      INT_NUMBER@5..6 "0"
    SEMICOLON@6..7 ";"
  EXPR_STMT@7..14
    BIN_EXPR@7..13
      PATH_EXPR@7..8
        PATH@7..8
          PATH_SEGMENT@7..8
            NAME_REF@7..8
              IDENT@7..8 "a"
      EQ@8..9 "="
      BIN_EXPR@9..13
        LITERAL@9..11
          INT_NUMBER@9..11 "10"
        PLUS@11..12 "+"
        LITERAL@12..13
          INT_NUMBER@12..13 "1"
    SEMICOLON@13..14 ";"
  PATH_EXPR@14..15
    PATH@14..15
      PATH_SEGMENT@14..15
        NAME_REF@14..15
          IDENT@14..15 "a""#,
    );
}

#[test]
fn test_match_literal() {
    parse_macro(
        r#"
    macro_rules! foo {
        ('(') => {
            fn foo() {}
        }
    }
"#,
    )
    .assert_expand_items("foo! ['('];", "fn foo () {}");
}

#[test]
fn test_parse_macro_def_simple() {
    cov_mark::check!(parse_macro_def_simple);

    parse_macro2(
        r#"
macro foo($id:ident) {
    fn $id() {}
}
"#,
    )
    .assert_expand_items("foo!(bar);", "fn bar () {}");
}

#[test]
fn test_parse_macro_def_rules() {
    cov_mark::check!(parse_macro_def_rules);

    parse_macro2(
        r#"
macro foo {
    ($id:ident) => {
        fn $id() {}
    }
}
"#,
    )
    .assert_expand_items("foo!(bar);", "fn bar () {}");
}

#[test]
fn test_path() {
    parse_macro(
        r#"
        macro_rules! foo {
            ($ i:path) => {
                fn foo() { let a = $ i; }
            }
        }
"#,
    )
    .assert_expand_items("foo! { foo }", "fn foo () {let a = foo ;}")
    .assert_expand_items(
        "foo! { bar::<u8>::baz::<u8> }",
        "fn foo () {let a = bar ::< u8 >:: baz ::< u8 > ;}",
    );
}

#[test]
fn test_two_paths() {
    parse_macro(
        r#"
        macro_rules! foo {
            ($ i:path, $ j:path) => {
                fn foo() { let a = $ i; let b = $j; }
            }
        }
"#,
    )
    .assert_expand_items("foo! { foo, bar }", "fn foo () {let a = foo ; let b = bar ;}");
}

#[test]
fn test_path_with_path() {
    parse_macro(
        r#"
        macro_rules! foo {
            ($ i:path) => {
                fn foo() { let a = $ i :: bar; }
            }
        }
"#,
    )
    .assert_expand_items("foo! { foo }", "fn foo () {let a = foo :: bar ;}");
}

#[test]
fn test_expr() {
    parse_macro(
        r#"
        macro_rules! foo {
            ($ i:expr) => {
                 fn bar() { $ i; }
            }
        }
"#,
    )
    .assert_expand_items(
        "foo! { 2 + 2 * baz(3).quux() }",
        "fn bar () {2 + 2 * baz (3) . quux () ;}",
    );
}

#[test]
fn test_last_expr() {
    parse_macro(
        r#"
        macro_rules! vec {
            ($($item:expr),*) => {
                {
                    let mut v = Vec::new();
                    $(
                        v.push($item);
                    )*
                    v
                }
            };
        }
"#,
    )
    .assert_expand_items(
        "vec!(1,2,3);",
        "{let mut v = Vec :: new () ; v . push (1) ; v . push (2) ; v . push (3) ; v}",
    );
}

#[test]
fn test_expr_with_attr() {
    parse_macro(
        r#"
macro_rules! m {
    ($a:expr) => {0}
}
"#,
    )
    .assert_expand_items("m!(#[allow(a)]())", "0");
}

#[test]
fn test_ty() {
    parse_macro(
        r#"
        macro_rules! foo {
            ($ i:ty) => (
                fn bar() -> $ i { unimplemented!() }
            )
        }
"#,
    )
    .assert_expand_items("foo! { Baz<u8> }", "fn bar () -> Baz < u8 > {unimplemented ! ()}");
}

#[test]
fn test_ty_with_complex_type() {
    parse_macro(
        r#"
        macro_rules! foo {
            ($ i:ty) => (
                fn bar() -> $ i { unimplemented!() }
            )
        }
"#,
    )
    // Reference lifetime struct with generic type
    .assert_expand_items(
        "foo! { &'a Baz<u8> }",
        "fn bar () -> & 'a Baz < u8 > {unimplemented ! ()}",
    )
    // extern "Rust" func type
    .assert_expand_items(
        r#"foo! { extern "Rust" fn() -> Ret }"#,
        r#"fn bar () -> extern "Rust" fn () -> Ret {unimplemented ! ()}"#,
    );
}

#[test]
fn test_pat_() {
    parse_macro(
        r#"
        macro_rules! foo {
            ($ i:pat) => { fn foo() { let $ i; } }
        }
"#,
    )
    .assert_expand_items("foo! { (a, b) }", "fn foo () {let (a , b) ;}");
}

#[test]
fn test_stmt() {
    parse_macro(
        r#"
        macro_rules! foo {
            ($ i:stmt) => (
                fn bar() { $ i; }
            )
        }
"#,
    )
    .assert_expand_items("foo! { 2 }", "fn bar () {2 ;}")
    .assert_expand_items("foo! { let a = 0 }", "fn bar () {let a = 0 ;}");
}

#[test]
fn test_single_item() {
    parse_macro(
        r#"
        macro_rules! foo {
            ($ i:item) => (
                $ i
            )
        }
"#,
    )
    .assert_expand_items("foo! {mod c {}}", "mod c {}");
}

#[test]
fn test_all_items() {
    parse_macro(
        r#"
        macro_rules! foo {
            ($ ($ i:item)*) => ($ (
                $ i
            )*)
        }
"#,
    ).
    assert_expand_items(
        r#"
        foo! {
            extern crate a;
            mod b;
            mod c {}
            use d;
            const E: i32 = 0;
            static F: i32 = 0;
            impl G {}
            struct H;
            enum I { Foo }
            trait J {}
            fn h() {}
            extern {}
            type T = u8;
        }
"#,
        r#"extern crate a ; mod b ; mod c {} use d ; const E : i32 = 0 ; static F : i32 = 0 ; impl G {} struct H ; enum I {Foo} trait J {} fn h () {} extern {} type T = u8 ;"#,
    );
}

#[test]
fn test_block() {
    parse_macro(
        r#"
        macro_rules! foo {
            ($ i:block) => { fn foo() $ i }
        }
"#,
    )
    .assert_expand_statements("foo! { { 1; } }", "fn foo () {1 ;}");
}

#[test]
fn test_meta() {
    parse_macro(
        r#"
        macro_rules! foo {
            ($ i:meta) => (
                #[$ i]
                fn bar() {}
            )
        }
"#,
    )
    .assert_expand_items(
        r#"foo! { cfg(target_os = "windows") }"#,
        r#"# [cfg (target_os = "windows")] fn bar () {}"#,
    )
    .assert_expand_items(r#"foo! { hello::world }"#, r#"# [hello :: world] fn bar () {}"#);
}

#[test]
fn test_meta_doc_comments() {
    parse_macro(
        r#"
        macro_rules! foo {
            ($(#[$ i:meta])+) => (
                $(#[$ i])+
                fn bar() {}
            )
        }
"#,
    ).
    assert_expand_items(
        r#"foo! {
            /// Single Line Doc 1
            /**
                MultiLines Doc
            */
        }"#,
        "# [doc = \" Single Line Doc 1\"] # [doc = \"\\\\n                MultiLines Doc\\\\n            \"] fn bar () {}",
    );
}

#[test]
fn test_meta_doc_comments_non_latin() {
    parse_macro(
        r#"
        macro_rules! foo {
            ($(#[$ i:meta])+) => (
                $(#[$ i])+
                fn bar() {}
            )
        }
"#,
    ).
    assert_expand_items(
        r#"foo! {
            /// 錦瑟無端五十弦，一弦一柱思華年。
            /**
                莊生曉夢迷蝴蝶，望帝春心託杜鵑。
            */
        }"#,
        "# [doc = \" 錦瑟無端五十弦，一弦一柱思華年。\"] # [doc = \"\\\\n                莊生曉夢迷蝴蝶，望帝春心託杜鵑。\\\\n            \"] fn bar () {}",
    );
}

#[test]
fn test_tt_block() {
    parse_macro(
        r#"
            macro_rules! foo {
                ($ i:tt) => { fn foo() $ i }
            }
    "#,
    )
    .assert_expand_items(r#"foo! { { 1; } }"#, r#"fn foo () {1 ;}"#);
}

#[test]
fn test_tt_group() {
    parse_macro(
        r#"
            macro_rules! foo {
                 ($($ i:tt)*) => { $($ i)* }
            }
    "#,
    )
    .assert_expand_items(r#"foo! { fn foo() {} }"#, r#"fn foo () {}"#);
}

#[test]
fn test_tt_composite() {
    parse_macro(
        r#"
            macro_rules! foo {
                 ($i:tt) => { 0 }
            }
    "#,
    )
    .assert_expand_items(r#"foo! { => }"#, r#"0"#);
}

#[test]
fn test_tt_composite2() {
    let node = parse_macro(
        r#"
            macro_rules! foo {
                ($($tt:tt)*) => { abs!(=> $($tt)*) }
            }
    "#,
    )
    .expand_items(r#"foo!{#}"#);

    let res = format!("{:#?}", &node);
    assert_eq_text!(
        r###"MACRO_ITEMS@0..10
  MACRO_CALL@0..10
    PATH@0..3
      PATH_SEGMENT@0..3
        NAME_REF@0..3
          IDENT@0..3 "abs"
    BANG@3..4 "!"
    TOKEN_TREE@4..10
      L_PAREN@4..5 "("
      EQ@5..6 "="
      R_ANGLE@6..7 ">"
      WHITESPACE@7..8 " "
      POUND@8..9 "#"
      R_PAREN@9..10 ")""###,
        res.trim()
    );
}

#[test]
fn test_tt_with_composite_without_space() {
    parse_macro(
        r#"
        macro_rules! foo {
            ($ op:tt, $j:path) => (
                0
            )
        }
"#,
    )
    // Test macro input without any spaces
    // See https://github.com/rust-analyzer/rust-analyzer/issues/6692
    .assert_expand_items("foo!(==,Foo::Bool)", "0");
}

#[test]
fn test_underscore() {
    parse_macro(
        r#"
            macro_rules! foo {
                 ($_:tt) => { 0 }
            }
    "#,
    )
    .assert_expand_items(r#"foo! { => }"#, r#"0"#);
}

#[test]
fn test_underscore_not_greedily() {
    parse_macro(
        r#"
macro_rules! q {
    ($($a:ident)* _) => {0};
}
"#,
    )
    // `_` overlaps with `$a:ident` but rustc matches it under the `_` token
    .assert_expand_items(r#"q![a b c d _]"#, r#"0"#);

    parse_macro(
        r#"
macro_rules! q {
    ($($a:expr => $b:ident)* _ => $c:expr) => {0};
}
"#,
    )
    // `_ => ou` overlaps with `$a:expr => $b:ident` but rustc matches it under `_ => $c:expr`
    .assert_expand_items(r#"q![a => b c => d _ => ou]"#, r#"0"#);
}

#[test]
fn test_underscore_as_type() {
    parse_macro(
        r#"
macro_rules! q {
    ($a:ty) => {0};
}
"#,
    )
    // Underscore is a type
    .assert_expand_items(r#"q![_]"#, r#"0"#);
}

#[test]
fn test_vertical_bar_with_pat() {
    parse_macro(
        r#"
            macro_rules! foo {
                 (| $pat:pat | ) => { 0 }
            }
    "#,
    )
    .assert_expand_items(r#"foo! { | x | }"#, r#"0"#);
}

#[test]
fn test_dollar_crate_lhs_is_not_meta() {
    parse_macro(
        r#"
macro_rules! foo {
    ($crate) => {};
    () => {0};
}
    "#,
    )
    .assert_expand_items(r#"foo!{}"#, r#"0"#);
}

#[test]
fn test_lifetime() {
    parse_macro(
        r#"
        macro_rules! foo {
              ($ lt:lifetime) => { struct Ref<$ lt>{ s: &$ lt str } }
        }
"#,
    )
    .assert_expand_items(r#"foo!{'a}"#, r#"struct Ref <'a > {s : &'a str}"#);
}

#[test]
fn test_literal() {
    parse_macro(
        r#"
        macro_rules! foo {
              ($ type:ty , $ lit:literal) => { const VALUE: $ type = $ lit;};
        }
"#,
    )
    .assert_expand_items(r#"foo!(u8,0);"#, r#"const VALUE : u8 = 0 ;"#);

    parse_macro(
        r#"
        macro_rules! foo {
              ($ type:ty , $ lit:literal) => { const VALUE: $ type = $ lit;};
        }
"#,
    )
    .assert_expand_items(r#"foo!(i32,-1);"#, r#"const VALUE : i32 = - 1 ;"#);
}

#[test]
fn test_boolean_is_ident() {
    parse_macro(
        r#"
        macro_rules! foo {
              ($lit0:literal, $lit1:literal) => { const VALUE: (bool,bool) = ($lit0,$lit1); };
        }
"#,
    )
    .assert_expand(
        r#"foo!(true,false);"#,
        r#"
SUBTREE $
  IDENT   const 14
  IDENT   VALUE 15
  PUNCH   : [alone] 16
  SUBTREE () 17
    IDENT   bool 18
    PUNCH   , [alone] 19
    IDENT   bool 20
  PUNCH   = [alone] 21
  SUBTREE () 22
    IDENT   true 29
    PUNCH   , [joint] 25
    IDENT   false 31
  PUNCH   ; [alone] 28
"#,
    );
}

#[test]
fn test_vis() {
    parse_macro(
        r#"
        macro_rules! foo {
              ($ vis:vis $ name:ident) => { $ vis fn $ name() {}};
        }
"#,
    )
    .assert_expand_items(r#"foo!(pub foo);"#, r#"pub fn foo () {}"#)
    // test optional cases
    .assert_expand_items(r#"foo!(foo);"#, r#"fn foo () {}"#);
}

#[test]
fn test_inner_macro_rules() {
    parse_macro(
        r#"
macro_rules! foo {
    ($a:ident, $b:ident, $c:tt) => {

        macro_rules! bar {
            ($bi:ident) => {
                fn $bi() -> u8 {$c}
            }
        }

        bar!($a);
        fn $b() -> u8 {$c}
    }
}
"#,
    ).
    assert_expand_items(
        r#"foo!(x,y, 1);"#,
        r#"macro_rules ! bar {($ bi : ident) => {fn $ bi () -> u8 {1}}} bar ! (x) ; fn y () -> u8 {1}"#,
    );
}

#[test]
fn test_expr_after_path_colons() {
    assert!(parse_macro(
        r#"
macro_rules! m {
    ($k:expr) => {
            f(K::$k);
       }
}
"#,
    )
    .expand_statements(r#"m!(C("0"))"#)
    .descendants()
    .find(|token| token.kind() == ERROR)
    .is_some());
}

#[test]
fn test_match_is_not_greedy() {
    parse_macro(
        r#"
macro_rules! foo {
    ($($i:ident $(,)*),*) => {};
}
"#,
    )
    .assert_expand_items(r#"foo!(a,b);"#, r#""#);
}

// The following tests are based on real world situations
#[test]
fn test_vec() {
    let fixture = parse_macro(
        r#"
         macro_rules! vec {
            ($($item:expr),*) => {
                {
                    let mut v = Vec::new();
                    $(
                        v.push($item);
                    )*
                    v
                }
            };
}
"#,
    );
    fixture
        .assert_expand_items(r#"vec!();"#, r#"{let mut v = Vec :: new () ; v}"#)
        .assert_expand_items(
            r#"vec![1u32,2];"#,
            r#"{let mut v = Vec :: new () ; v . push (1u32) ; v . push (2) ; v}"#,
        );

    let tree = fixture.expand_expr(r#"vec![1u32,2];"#);

    assert_eq!(
        format!("{:#?}", tree).trim(),
        r#"BLOCK_EXPR@0..45
  L_CURLY@0..1 "{"
  LET_STMT@1..20
    LET_KW@1..4 "let"
    IDENT_PAT@4..8
      MUT_KW@4..7 "mut"
      NAME@7..8
        IDENT@7..8 "v"
    EQ@8..9 "="
    CALL_EXPR@9..19
      PATH_EXPR@9..17
        PATH@9..17
          PATH@9..12
            PATH_SEGMENT@9..12
              NAME_REF@9..12
                IDENT@9..12 "Vec"
          COLON2@12..14 "::"
          PATH_SEGMENT@14..17
            NAME_REF@14..17
              IDENT@14..17 "new"
      ARG_LIST@17..19
        L_PAREN@17..18 "("
        R_PAREN@18..19 ")"
    SEMICOLON@19..20 ";"
  EXPR_STMT@20..33
    METHOD_CALL_EXPR@20..32
      PATH_EXPR@20..21
        PATH@20..21
          PATH_SEGMENT@20..21
            NAME_REF@20..21
              IDENT@20..21 "v"
      DOT@21..22 "."
      NAME_REF@22..26
        IDENT@22..26 "push"
      ARG_LIST@26..32
        L_PAREN@26..27 "("
        LITERAL@27..31
          INT_NUMBER@27..31 "1u32"
        R_PAREN@31..32 ")"
    SEMICOLON@32..33 ";"
  EXPR_STMT@33..43
    METHOD_CALL_EXPR@33..42
      PATH_EXPR@33..34
        PATH@33..34
          PATH_SEGMENT@33..34
            NAME_REF@33..34
              IDENT@33..34 "v"
      DOT@34..35 "."
      NAME_REF@35..39
        IDENT@35..39 "push"
      ARG_LIST@39..42
        L_PAREN@39..40 "("
        LITERAL@40..41
          INT_NUMBER@40..41 "2"
        R_PAREN@41..42 ")"
    SEMICOLON@42..43 ";"
  PATH_EXPR@43..44
    PATH@43..44
      PATH_SEGMENT@43..44
        NAME_REF@43..44
          IDENT@43..44 "v"
  R_CURLY@44..45 "}""#
    );
}

#[test]
fn test_winapi_struct() {
    // from https://github.com/retep998/winapi-rs/blob/a7ef2bca086aae76cf6c4ce4c2552988ed9798ad/src/macros.rs#L366

    parse_macro(
        r#"
macro_rules! STRUCT {
    ($(#[$attrs:meta])* struct $name:ident {
        $($field:ident: $ftype:ty,)+
    }) => (
        #[repr(C)] #[derive(Copy)] $(#[$attrs])*
        pub struct $name {
            $(pub $field: $ftype,)+
        }
        impl Clone for $name {
            #[inline]
            fn clone(&self) -> $name { *self }
        }
        #[cfg(feature = "impl-default")]
        impl Default for $name {
            #[inline]
            fn default() -> $name { unsafe { $crate::_core::mem::zeroed() } }
        }
    );
}
"#,
    ).
    // from https://github.com/retep998/winapi-rs/blob/a7ef2bca086aae76cf6c4ce4c2552988ed9798ad/src/shared/d3d9caps.rs
    assert_expand_items(r#"STRUCT!{struct D3DVSHADERCAPS2_0 {Caps: u8,}}"#,
        "# [repr (C)] # [derive (Copy)] pub struct D3DVSHADERCAPS2_0 {pub Caps : u8 ,} impl Clone for D3DVSHADERCAPS2_0 {# [inline] fn clone (& self) -> D3DVSHADERCAPS2_0 {* self}} # [cfg (feature = \"impl-default\")] impl Default for D3DVSHADERCAPS2_0 {# [inline] fn default () -> D3DVSHADERCAPS2_0 {unsafe {$crate :: _core :: mem :: zeroed ()}}}"
    )
    .assert_expand_items(r#"STRUCT!{#[cfg_attr(target_arch = "x86", repr(packed))] struct D3DCONTENTPROTECTIONCAPS {Caps : u8 ,}}"#,
        "# [repr (C)] # [derive (Copy)] # [cfg_attr (target_arch = \"x86\" , repr (packed))] pub struct D3DCONTENTPROTECTIONCAPS {pub Caps : u8 ,} impl Clone for D3DCONTENTPROTECTIONCAPS {# [inline] fn clone (& self) -> D3DCONTENTPROTECTIONCAPS {* self}} # [cfg (feature = \"impl-default\")] impl Default for D3DCONTENTPROTECTIONCAPS {# [inline] fn default () -> D3DCONTENTPROTECTIONCAPS {unsafe {$crate :: _core :: mem :: zeroed ()}}}"
    );
}

#[test]
fn test_int_base() {
    parse_macro(
        r#"
macro_rules! int_base {
    ($Trait:ident for $T:ident as $U:ident -> $Radix:ident) => {
        #[stable(feature = "rust1", since = "1.0.0")]
        impl fmt::$Trait for $T {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                $Radix.fmt_int(*self as $U, f)
            }
        }
    }
}
"#,
    ).assert_expand_items(r#" int_base!{Binary for isize as usize -> Binary}"#,
        "# [stable (feature = \"rust1\" , since = \"1.0.0\")] impl fmt ::Binary for isize {fn fmt (& self , f : & mut fmt :: Formatter < \'_ >) -> fmt :: Result {Binary . fmt_int (* self as usize , f)}}"
    );
}

#[test]
fn test_generate_pattern_iterators() {
    // from https://github.com/rust-lang/rust/blob/316a391dcb7d66dc25f1f9a4ec9d368ef7615005/src/libcore/str/mod.rs
    parse_macro(
        r#"
macro_rules! generate_pattern_iterators {
        { double ended; with $(#[$common_stability_attribute:meta])*,
                           $forward_iterator:ident,
                           $reverse_iterator:ident, $iterty:ty
        } => {
            fn foo(){}
        }
}
"#,
    ).assert_expand_items(
        r#"generate_pattern_iterators ! ( double ended ; with # [ stable ( feature = "rust1" , since = "1.0.0" ) ] , Split , RSplit , & 'a str );"#,
        "fn foo () {}",
    );
}

#[test]
fn test_impl_fn_for_zst() {
    // from https://github.com/rust-lang/rust/blob/5d20ff4d2718c820632b38c1e49d4de648a9810b/src/libcore/internal_macros.rs
    parse_macro(
        r#"
macro_rules! impl_fn_for_zst  {
        {  $( $( #[$attr: meta] )*
        struct $Name: ident impl$( <$( $lifetime : lifetime ),+> )? Fn =
            |$( $arg: ident: $ArgTy: ty ),*| -> $ReturnTy: ty
$body: block; )+
        } => {
           $(
            $( #[$attr] )*
            struct $Name;

            impl $( <$( $lifetime ),+> )? Fn<($( $ArgTy, )*)> for $Name {
                #[inline]
                extern "rust-call" fn call(&self, ($( $arg, )*): ($( $ArgTy, )*)) -> $ReturnTy {
                    $body
                }
            }

            impl $( <$( $lifetime ),+> )? FnMut<($( $ArgTy, )*)> for $Name {
                #[inline]
                extern "rust-call" fn call_mut(
                    &mut self,
                    ($( $arg, )*): ($( $ArgTy, )*)
                ) -> $ReturnTy {
                    Fn::call(&*self, ($( $arg, )*))
                }
            }

            impl $( <$( $lifetime ),+> )? FnOnce<($( $ArgTy, )*)> for $Name {
                type Output = $ReturnTy;

                #[inline]
                extern "rust-call" fn call_once(self, ($( $arg, )*): ($( $ArgTy, )*)) -> $ReturnTy {
                    Fn::call(&self, ($( $arg, )*))
                }
            }
        )+
}
        }
"#,
    ).assert_expand_items(r#"
impl_fn_for_zst !   {
     # [ derive ( Clone ) ]
     struct   CharEscapeDebugContinue   impl   Fn   =   | c :   char |   ->   char :: EscapeDebug   {
         c . escape_debug_ext ( false )
     } ;

     # [ derive ( Clone ) ]
     struct   CharEscapeUnicode   impl   Fn   =   | c :   char |   ->   char :: EscapeUnicode   {
         c . escape_unicode ( )
     } ;
     # [ derive ( Clone ) ]
     struct   CharEscapeDefault   impl   Fn   =   | c :   char |   ->   char :: EscapeDefault   {
         c . escape_default ( )
     } ;
 }
"#,
        "# [derive (Clone)] struct CharEscapeDebugContinue ; impl Fn < (char ,) > for CharEscapeDebugContinue {# [inline] extern \"rust-call\" fn call (& self , (c ,) : (char ,)) -> char :: EscapeDebug {{c . escape_debug_ext (false)}}} impl FnMut < (char ,) > for CharEscapeDebugContinue {# [inline] extern \"rust-call\" fn call_mut (& mut self , (c ,) : (char ,)) -> char :: EscapeDebug {Fn :: call (&* self , (c ,))}} impl FnOnce < (char ,) > for CharEscapeDebugContinue {type Output = char :: EscapeDebug ; # [inline] extern \"rust-call\" fn call_once (self , (c ,) : (char ,)) -> char :: EscapeDebug {Fn :: call (& self , (c ,))}} # [derive (Clone)] struct CharEscapeUnicode ; impl Fn < (char ,) > for CharEscapeUnicode {# [inline] extern \"rust-call\" fn call (& self , (c ,) : (char ,)) -> char :: EscapeUnicode {{c . escape_unicode ()}}} impl FnMut < (char ,) > for CharEscapeUnicode {# [inline] extern \"rust-call\" fn call_mut (& mut self , (c ,) : (char ,)) -> char :: EscapeUnicode {Fn :: call (&* self , (c ,))}} impl FnOnce < (char ,) > for CharEscapeUnicode {type Output = char :: EscapeUnicode ; # [inline] extern \"rust-call\" fn call_once (self , (c ,) : (char ,)) -> char :: EscapeUnicode {Fn :: call (& self , (c ,))}} # [derive (Clone)] struct CharEscapeDefault ; impl Fn < (char ,) > for CharEscapeDefault {# [inline] extern \"rust-call\" fn call (& self , (c ,) : (char ,)) -> char :: EscapeDefault {{c . escape_default ()}}} impl FnMut < (char ,) > for CharEscapeDefault {# [inline] extern \"rust-call\" fn call_mut (& mut self , (c ,) : (char ,)) -> char :: EscapeDefault {Fn :: call (&* self , (c ,))}} impl FnOnce < (char ,) > for CharEscapeDefault {type Output = char :: EscapeDefault ; # [inline] extern \"rust-call\" fn call_once (self , (c ,) : (char ,)) -> char :: EscapeDefault {Fn :: call (& self , (c ,))}}"
    );
}

#[test]
fn test_impl_nonzero_fmt() {
    // from https://github.com/rust-lang/rust/blob/316a391dcb7d66dc25f1f9a4ec9d368ef7615005/src/libcore/num/mod.rs#L12
    parse_macro(
        r#"
        macro_rules! impl_nonzero_fmt {
            ( #[$stability: meta] ( $( $Trait: ident ),+ ) for $Ty: ident ) => {
                fn foo () {}
            }
        }
"#,
    ).assert_expand_items(
        r#"impl_nonzero_fmt! { # [stable(feature= "nonzero",since="1.28.0")] (Debug,Display,Binary,Octal,LowerHex,UpperHex) for NonZeroU8}"#,
        "fn foo () {}",
    );
}

#[test]
fn test_cfg_if_items() {
    // from https://github.com/rust-lang/rust/blob/33fe1131cadba69d317156847be9a402b89f11bb/src/libstd/macros.rs#L986
    parse_macro(
        r#"
        macro_rules! __cfg_if_items {
            (($($not:meta,)*) ; ) => {};
            (($($not:meta,)*) ; ( ($($m:meta),*) ($($it:item)*) ), $($rest:tt)*) => {
                 __cfg_if_items! { ($($not,)* $($m,)*) ; $($rest)* }
            }
        }
"#,
    ).assert_expand_items(
        r#"__cfg_if_items ! { ( rustdoc , ) ; ( ( ) ( # [ cfg ( any ( target_os = "redox" , unix ) ) ] # [ stable ( feature = "rust1" , since = "1.0.0" ) ] pub use sys :: ext as unix ; # [ cfg ( windows ) ] # [ stable ( feature = "rust1" , since = "1.0.0" ) ] pub use sys :: ext as windows ; # [ cfg ( any ( target_os = "linux" , target_os = "l4re" ) ) ] pub mod linux ; ) ) , }"#,
        "__cfg_if_items ! {(rustdoc ,) ;}",
    );
}

#[test]
fn test_cfg_if_main() {
    // from https://github.com/rust-lang/rust/blob/3d211248393686e0f73851fc7548f6605220fbe1/src/libpanic_unwind/macros.rs#L9
    parse_macro(
        r#"
        macro_rules! cfg_if {
            ($(
                if #[cfg($($meta:meta),*)] { $($it:item)* }
            ) else * else {
                $($it2:item)*
            }) => {
                __cfg_if_items! {
                    () ;
                    $( ( ($($meta),*) ($($it)*) ), )*
                    ( () ($($it2)*) ),
                }
            };

            // Internal macro to Apply a cfg attribute to a list of items
            (@__apply $m:meta, $($it:item)*) => {
                $(#[$m] $it)*
            };
        }
"#,
    ).assert_expand_items(r#"
cfg_if !   {
     if   # [ cfg ( target_env   =   "msvc" ) ]   {
         // no extra unwinder support needed
     }   else   if   # [ cfg ( all ( target_arch   =   "wasm32" ,   not ( target_os   =   "emscripten" ) ) ) ]   {
         // no unwinder on the system!
     }   else   {
         mod   libunwind ;
         pub   use   libunwind :: * ;
     }
 }
"#,
        "__cfg_if_items ! {() ; ((target_env = \"msvc\") ()) , ((all (target_arch = \"wasm32\" , not (target_os = \"emscripten\"))) ()) , (() (mod libunwind ; pub use libunwind :: * ;)) ,}"
    ).assert_expand_items(
        r#"
cfg_if ! { @ __apply cfg ( all ( not ( any ( not ( any ( target_os = "solaris" , target_os = "illumos" ) ) ) ) ) ) , }
"#,
        "",
    );
}

#[test]
fn test_proptest_arbitrary() {
    // from https://github.com/AltSysrq/proptest/blob/d1c4b049337d2f75dd6f49a095115f7c532e5129/proptest/src/arbitrary/macros.rs#L16
    parse_macro(
        r#"
macro_rules! arbitrary {
    ([$($bounds : tt)*] $typ: ty, $strat: ty, $params: ty;
        $args: ident => $logic: expr) => {
        impl<$($bounds)*> $crate::arbitrary::Arbitrary for $typ {
            type Parameters = $params;
            type Strategy = $strat;
            fn arbitrary_with($args: Self::Parameters) -> Self::Strategy {
                $logic
            }
        }
    };

}"#,
    ).assert_expand_items(r#"arbitrary !   ( [ A : Arbitrary ]
        Vec < A > ,
        VecStrategy < A :: Strategy > ,
        RangedParams1 < A :: Parameters > ;
        args =>   { let product_unpack !   [ range , a ] = args ; vec ( any_with :: < A >   ( a ) , range ) }
    ) ;"#,
    "impl <A : Arbitrary > $crate :: arbitrary :: Arbitrary for Vec < A > {type Parameters = RangedParams1 < A :: Parameters > ; type Strategy = VecStrategy < A :: Strategy > ; fn arbitrary_with (args : Self :: Parameters) -> Self :: Strategy {{let product_unpack ! [range , a] = args ; vec (any_with :: < A > (a) , range)}}}"
    );
}

#[test]
fn test_old_ridl() {
    // This is from winapi 2.8, which do not have a link from github
    //
    let expanded = parse_macro(
        r#"
#[macro_export]
macro_rules! RIDL {
    (interface $interface:ident ($vtbl:ident) : $pinterface:ident ($pvtbl:ident)
        {$(
            fn $method:ident(&mut self $(,$p:ident : $t:ty)*) -> $rtr:ty
        ),+}
    ) => {
        impl $interface {
            $(pub unsafe fn $method(&mut self) -> $rtr {
                ((*self.lpVtbl).$method)(self $(,$p)*)
            })+
        }
    };
}"#,
    ).expand_tt(r#"
    RIDL!{interface ID3D11Asynchronous(ID3D11AsynchronousVtbl): ID3D11DeviceChild(ID3D11DeviceChildVtbl) {
        fn GetDataSize(&mut self) -> UINT
    }}"#);

    assert_eq!(expanded.to_string(), "impl ID3D11Asynchronous {pub unsafe fn GetDataSize (& mut self) -> UINT {((* self . lpVtbl) .GetDataSize) (self)}}");
}

#[test]
fn test_quick_error() {
    let expanded = parse_macro(
        r#"
macro_rules! quick_error {

 (SORT [enum $name:ident $( #[$meta:meta] )*]
        items [$($( #[$imeta:meta] )*
                  => $iitem:ident: $imode:tt [$( $ivar:ident: $ityp:ty ),*]
                                {$( $ifuncs:tt )*} )* ]
        buf [ ]
        queue [ ]
    ) => {
        quick_error!(ENUMINITION [enum $name $( #[$meta] )*]
            body []
            queue [$(
                $( #[$imeta] )*
                =>
                $iitem: $imode [$( $ivar: $ityp ),*]
            )*]
        );
};

}
"#,
    )
    .expand_tt(
        r#"
quick_error ! (SORT [enum Wrapped # [derive (Debug)]] items [
        => One : UNIT [] {}
        => Two : TUPLE [s :String] {display ("two: {}" , s) from ()}
    ] buf [] queue []) ;
"#,
    );

    assert_eq!(expanded.to_string(), "quick_error ! (ENUMINITION [enum Wrapped # [derive (Debug)]] body [] queue [=> One : UNIT [] => Two : TUPLE [s : String]]) ;");
}

#[test]
fn test_empty_repeat_vars_in_empty_repeat_vars() {
    parse_macro(
        r#"
macro_rules! delegate_impl {
    ([$self_type:ident, $self_wrap:ty, $self_map:ident]
     pub trait $name:ident $(: $sup:ident)* $(+ $more_sup:ident)* {

        $(
        @escape [type $assoc_name_ext:ident]
        )*
        $(
        @section type
        $(
            $(#[$_assoc_attr:meta])*
            type $assoc_name:ident $(: $assoc_bound:ty)*;
        )+
        )*
        $(
        @section self
        $(
            $(#[$_method_attr:meta])*
            fn $method_name:ident(self $(: $self_selftype:ty)* $(,$marg:ident : $marg_ty:ty)*) -> $mret:ty;
        )+
        )*
        $(
        @section nodelegate
        $($tail:tt)*
        )*
    }) => {
        impl<> $name for $self_wrap where $self_type: $name {
            $(
            $(
                fn $method_name(self $(: $self_selftype)* $(,$marg: $marg_ty)*) -> $mret {
                    $self_map!(self).$method_name($($marg),*)
                }
            )*
            )*
        }
    }
}
"#,
    ).assert_expand_items(
        r#"delegate_impl ! {[G , & 'a mut G , deref] pub trait Data : GraphBase {@ section type type NodeWeight ;}}"#,
        "impl <> Data for & \'a mut G where G : Data {}",
    );
}

#[test]
fn expr_interpolation() {
    let expanded = parse_macro(
        r#"
        macro_rules! id {
            ($expr:expr) => {
                map($expr)
            }
        }
        "#,
    )
    .expand_expr("id!(x + foo);");

    assert_eq!(expanded.to_string(), "map(x+foo)");
}

#[test]
fn test_issue_2520() {
    let macro_fixture = parse_macro(
        r#"
        macro_rules! my_macro {
            {
                ( $(
                    $( [] $sname:ident : $stype:ty  )?
                    $( [$expr:expr] $nname:ident : $ntype:ty  )?
                ),* )
            } => {
                Test {
                    $(
                        $( $sname, )?
                    )*
                }
            };
        }
    "#,
    );

    macro_fixture.assert_expand_items(
        r#"my_macro ! {
            ([] p1 : u32 , [|_| S0K0] s : S0K0 , [] k0 : i32)
        }"#,
        "Test {p1 , k0 ,}",
    );
}

#[test]
fn test_issue_3861() {
    let macro_fixture = parse_macro(
        r#"
        macro_rules! rgb_color {
            ($p:expr, $t: ty) => {
                pub fn new() {
                    let _ = 0 as $t << $p;
                }
            };
        }
    "#,
    );

    macro_fixture.expand_items(r#"rgb_color!(8 + 8, u32);"#);
}

#[test]
fn test_repeat_bad_var() {
    // FIXME: the second rule of the macro should be removed and an error about
    // `$( $c )+` raised
    parse_macro(
        r#"
        macro_rules! foo {
            ($( $b:ident )+) => {
                $( $c )+
            };
            ($( $b:ident )+) => {
                $( $b )+
            }
        }
    "#,
    )
    .assert_expand_items("foo!(b0 b1);", "b0 b1");
}

#[test]
fn test_no_space_after_semi_colon() {
    let expanded = parse_macro(
        r#"
        macro_rules! with_std { ($($i:item)*) => ($(#[cfg(feature = "std")]$i)*) }
    "#,
    )
    .expand_items(r#"with_std! {mod m;mod f;}"#);

    let dump = format!("{:#?}", expanded);
    assert_eq_text!(
        r###"MACRO_ITEMS@0..52
  MODULE@0..26
    ATTR@0..21
      POUND@0..1 "#"
      L_BRACK@1..2 "["
      PATH@2..5
        PATH_SEGMENT@2..5
          NAME_REF@2..5
            IDENT@2..5 "cfg"
      TOKEN_TREE@5..20
        L_PAREN@5..6 "("
        IDENT@6..13 "feature"
        EQ@13..14 "="
        STRING@14..19 "\"std\""
        R_PAREN@19..20 ")"
      R_BRACK@20..21 "]"
    MOD_KW@21..24 "mod"
    NAME@24..25
      IDENT@24..25 "m"
    SEMICOLON@25..26 ";"
  MODULE@26..52
    ATTR@26..47
      POUND@26..27 "#"
      L_BRACK@27..28 "["
      PATH@28..31
        PATH_SEGMENT@28..31
          NAME_REF@28..31
            IDENT@28..31 "cfg"
      TOKEN_TREE@31..46
        L_PAREN@31..32 "("
        IDENT@32..39 "feature"
        EQ@39..40 "="
        STRING@40..45 "\"std\""
        R_PAREN@45..46 ")"
      R_BRACK@46..47 "]"
    MOD_KW@47..50 "mod"
    NAME@50..51
      IDENT@50..51 "f"
    SEMICOLON@51..52 ";""###,
        dump.trim()
    );
}

// https://github.com/rust-lang/rust/blob/master/src/test/ui/issues/issue-57597.rs
#[test]
fn test_rustc_issue_57597() {
    fn test_error(fixture: &str) {
        assert_eq!(parse_macro_error(fixture), ParseError::RepetitionEmptyTokenTree);
    }

    test_error("macro_rules! foo { ($($($i:ident)?)+) => {}; }");
    test_error("macro_rules! foo { ($($($i:ident)?)*) => {}; }");
    test_error("macro_rules! foo { ($($($i:ident)?)?) => {}; }");
    test_error("macro_rules! foo { ($($($($i:ident)?)?)?) => {}; }");
    test_error("macro_rules! foo { ($($($($i:ident)*)?)?) => {}; }");
    test_error("macro_rules! foo { ($($($($i:ident)?)*)?) => {}; }");
    test_error("macro_rules! foo { ($($($($i:ident)?)?)*) => {}; }");
    test_error("macro_rules! foo { ($($($($i:ident)*)*)?) => {}; }");
    test_error("macro_rules! foo { ($($($($i:ident)?)*)*) => {}; }");
    test_error("macro_rules! foo { ($($($($i:ident)?)*)+) => {}; }");
    test_error("macro_rules! foo { ($($($($i:ident)+)?)*) => {}; }");
    test_error("macro_rules! foo { ($($($($i:ident)+)*)?) => {}; }");
}

#[test]
fn test_expand_bad_literal() {
    parse_macro(
        r#"
        macro_rules! foo { ($i:literal) => {}; }
    "#,
    )
    .assert_expand_err(r#"foo!(&k");"#, &ExpandError::BindingError("".into()));
}

#[test]
fn test_empty_comments() {
    parse_macro(
        r#"
        macro_rules! one_arg_macro { ($fmt:expr) => (); }
    "#,
    )
    .assert_expand_err(
        r#"one_arg_macro!(/**/)"#,
        &ExpandError::BindingError("expected Expr".into()),
    );
}
