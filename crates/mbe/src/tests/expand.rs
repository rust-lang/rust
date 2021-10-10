use ::parser::ParserEntryPoint;
use syntax::{SyntaxKind::IDENT, T};
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

    let (node, token_map) = token_tree_to_syntax_node(&expanded, ParserEntryPoint::Items).unwrap();
    let content = node.syntax_node().to_string();

    let get_text = |id, kind| -> String {
        content[token_map.first_range_by_token(id, kind).unwrap()].to_string()
    };

    assert_eq!(expanded.token_trees.len(), 4);
    // {($e:ident) => { fn $e() {} }}
    // 012345      67 8 9  T12  3

    assert_eq!(get_text(tt::TokenId(9), IDENT), "fn");
    assert_eq!(get_text(tt::TokenId(12), T!['(']), "(");
    assert_eq!(get_text(tt::TokenId(13), T!['{']), "{");
}

fn to_subtree(tt: &tt::TokenTree) -> &tt::Subtree {
    if let tt::TokenTree::Subtree(subtree) = tt {
        return subtree;
    }
    unreachable!("It is not a subtree");
}

fn to_punct(tt: &tt::TokenTree) -> &tt::Punct {
    if let tt::TokenTree::Leaf(tt::Leaf::Punct(lit)) = tt {
        return lit;
    }
    unreachable!("It is not a Punct");
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
      META@2..20
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
      META@28..46
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
