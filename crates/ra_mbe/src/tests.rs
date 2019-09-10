use ra_syntax::{ast, AstNode, NodeOrToken};
use test_utils::assert_eq_text;

use super::*;

// Good first issue (although a slightly challenging one):
//
// * Pick a random test from here
//   https://github.com/intellij-rust/intellij-rust/blob/c4e9feee4ad46e7953b1948c112533360b6087bb/src/test/kotlin/org/rust/lang/core/macros/RsMacroExpansionTest.kt
// * Port the test to rust and add it to this module
// * Make it pass :-)

#[test]
fn test_convert_tt() {
    let macro_definition = r#"
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
"#;

    let macro_invocation = r#"
impl_froms!(TokenTree: Leaf, Subtree);
"#;

    let source_file = ast::SourceFile::parse(macro_definition).ok().unwrap();
    let macro_definition =
        source_file.syntax().descendants().find_map(ast::MacroCall::cast).unwrap();

    let source_file = ast::SourceFile::parse(macro_invocation).ok().unwrap();
    let macro_invocation =
        source_file.syntax().descendants().find_map(ast::MacroCall::cast).unwrap();

    let (definition_tt, _) = ast_to_token_tree(&macro_definition.token_tree().unwrap()).unwrap();
    let (invocation_tt, _) = ast_to_token_tree(&macro_invocation.token_tree().unwrap()).unwrap();
    let rules = crate::MacroRules::parse(&definition_tt).unwrap();
    let expansion = rules.expand(&invocation_tt).unwrap();
    assert_eq!(
        expansion.to_string(),
        "impl From <Leaf > for TokenTree {fn from (it : Leaf) -> TokenTree {TokenTree ::Leaf (it)}} \
         impl From <Subtree > for TokenTree {fn from (it : Subtree) -> TokenTree {TokenTree ::Subtree (it)}}"
    )
}

pub(crate) fn create_rules(macro_definition: &str) -> MacroRules {
    let source_file = ast::SourceFile::parse(macro_definition).ok().unwrap();
    let macro_definition =
        source_file.syntax().descendants().find_map(ast::MacroCall::cast).unwrap();

    let (definition_tt, _) = ast_to_token_tree(&macro_definition.token_tree().unwrap()).unwrap();
    crate::MacroRules::parse(&definition_tt).unwrap()
}

pub(crate) fn expand(rules: &MacroRules, invocation: &str) -> tt::Subtree {
    let source_file = ast::SourceFile::parse(invocation).ok().unwrap();
    let macro_invocation =
        source_file.syntax().descendants().find_map(ast::MacroCall::cast).unwrap();

    let (invocation_tt, _) = ast_to_token_tree(&macro_invocation.token_tree().unwrap()).unwrap();

    rules.expand(&invocation_tt).unwrap()
}

pub(crate) fn expand_to_items(rules: &MacroRules, invocation: &str) -> ast::MacroItems {
    let expanded = expand(rules, invocation);
    token_tree_to_macro_items(&expanded).unwrap().tree()
}

#[allow(unused)]
pub(crate) fn expand_to_stmts(rules: &MacroRules, invocation: &str) -> ast::MacroStmts {
    let expanded = expand(rules, invocation);
    token_tree_to_macro_stmts(&expanded).unwrap().tree()
}

pub(crate) fn expand_to_expr(rules: &MacroRules, invocation: &str) -> ast::Expr {
    let expanded = expand(rules, invocation);
    token_tree_to_expr(&expanded).unwrap().tree()
}

pub(crate) fn text_to_tokentree(text: &str) -> tt::Subtree {
    // wrap the given text to a macro call
    let wrapped = format!("wrap_macro!( {} )", text);
    let wrapped = ast::SourceFile::parse(&wrapped);
    let wrapped = wrapped.tree().syntax().descendants().find_map(ast::TokenTree::cast).unwrap();
    let mut wrapped = ast_to_token_tree(&wrapped).unwrap().0;
    wrapped.delimiter = tt::Delimiter::None;

    wrapped
}

pub(crate) enum MacroKind {
    Items,
    Stmts,
}

use ra_syntax::WalkEvent;

pub fn debug_dump_ignore_spaces(node: &ra_syntax::SyntaxNode) -> String {
    use std::fmt::Write;

    let mut level = 0;
    let mut buf = String::new();
    macro_rules! indent {
        () => {
            for _ in 0..level {
                buf.push_str("  ");
            }
        };
    }

    for event in node.preorder_with_tokens() {
        match event {
            WalkEvent::Enter(element) => {
                match element {
                    NodeOrToken::Node(node) => {
                        indent!();
                        writeln!(buf, "{:?}", node.kind()).unwrap();
                    }
                    NodeOrToken::Token(token) => match token.kind() {
                        ra_syntax::SyntaxKind::WHITESPACE => {}
                        _ => {
                            indent!();
                            writeln!(buf, "{:?}", token.kind()).unwrap();
                        }
                    },
                }
                level += 1;
            }
            WalkEvent::Leave(_) => level -= 1,
        }
    }

    buf
}

pub(crate) fn assert_expansion(
    kind: MacroKind,
    rules: &MacroRules,
    invocation: &str,
    expected: &str,
) -> tt::Subtree {
    let expanded = expand(rules, invocation);
    assert_eq!(expanded.to_string(), expected);

    let expected = expected.replace("$crate", "C_C__C");

    // wrap the given text to a macro call
    let expected = text_to_tokentree(&expected);
    let (expanded_tree, expected_tree) = match kind {
        MacroKind::Items => {
            let expanded_tree = token_tree_to_macro_items(&expanded).unwrap().tree();
            let expected_tree = token_tree_to_macro_items(&expected).unwrap().tree();

            (
                debug_dump_ignore_spaces(expanded_tree.syntax()).trim().to_string(),
                debug_dump_ignore_spaces(expected_tree.syntax()).trim().to_string(),
            )
        }

        MacroKind::Stmts => {
            let expanded_tree = token_tree_to_macro_stmts(&expanded).unwrap().tree();
            let expected_tree = token_tree_to_macro_stmts(&expected).unwrap().tree();

            (
                debug_dump_ignore_spaces(expanded_tree.syntax()).trim().to_string(),
                debug_dump_ignore_spaces(expected_tree.syntax()).trim().to_string(),
            )
        }
    };

    let expected_tree = expected_tree.replace("C_C__C", "$crate");
    assert_eq!(
        expanded_tree, expected_tree,
        "\nleft:\n{}\nright:\n{}",
        expanded_tree, expected_tree,
    );

    expanded
}

#[test]
fn test_fail_match_pattern_by_first_token() {
    let rules = create_rules(
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
    );

    assert_expansion(MacroKind::Items, &rules, "foo! { foo }", "mod foo {}");
    assert_expansion(MacroKind::Items, &rules, "foo! { = bar }", "fn bar () {}");
    assert_expansion(MacroKind::Items, &rules, "foo! { + Baz }", "struct Baz ;");
}

#[test]
fn test_fail_match_pattern_by_last_token() {
    let rules = create_rules(
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
    );

    assert_expansion(MacroKind::Items, &rules, "foo! { foo }", "mod foo {}");
    assert_expansion(MacroKind::Items, &rules, "foo! { bar = }", "fn bar () {}");
    assert_expansion(MacroKind::Items, &rules, "foo! { Baz + }", "struct Baz ;");
}

#[test]
fn test_fail_match_pattern_by_word_token() {
    let rules = create_rules(
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
    );

    assert_expansion(MacroKind::Items, &rules, "foo! { foo }", "mod foo {}");
    assert_expansion(MacroKind::Items, &rules, "foo! { spam bar }", "fn bar () {}");
    assert_expansion(MacroKind::Items, &rules, "foo! { eggs Baz }", "struct Baz ;");
}

#[test]
fn test_match_group_pattern_by_separator_token() {
    let rules = create_rules(
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
    );

    assert_expansion(MacroKind::Items, &rules, "foo! { foo, bar }", "mod foo {} mod bar {}");
    assert_expansion(MacroKind::Items, &rules, "foo! { foo# bar }", "fn foo () {} fn bar () {}");
    assert_expansion(MacroKind::Items, &rules, "foo! { Foo,# Bar }", "struct Foo ; struct Bar ;");
}

#[test]
fn test_match_group_pattern_with_multiple_defs() {
    let rules = create_rules(
        r#"
        macro_rules! foo {
            ($ ($ i:ident),*) => ( struct Bar { $ (
                fn $ i {}
            )*} );
        }
"#,
    );

    assert_expansion(
        MacroKind::Items,
        &rules,
        "foo! { foo, bar }",
        "struct Bar {fn foo {} fn bar {}}",
    );
}

#[test]
fn test_match_group_pattern_with_multiple_statement() {
    let rules = create_rules(
        r#"
        macro_rules! foo {
            ($ ($ i:ident),*) => ( fn baz { $ (
                $ i ();
            )*} );
        }
"#,
    );

    assert_expansion(MacroKind::Items, &rules, "foo! { foo, bar }", "fn baz {foo () ; bar () ;}");
}

#[test]
fn test_match_group_pattern_with_multiple_statement_without_semi() {
    let rules = create_rules(
        r#"
        macro_rules! foo {
            ($ ($ i:ident),*) => ( fn baz { $ (
                $i()
            );*} );
        }
"#,
    );

    assert_expansion(MacroKind::Items, &rules, "foo! { foo, bar }", "fn baz {foo () ;bar ()}");
}

#[test]
fn test_match_group_empty_fixed_token() {
    let rules = create_rules(
        r#"
        macro_rules! foo {
            ($ ($ i:ident)* #abc) => ( fn baz { $ (
                $ i ();
            )*} );
        }
"#,
    );

    assert_expansion(MacroKind::Items, &rules, "foo! {#abc}", "fn baz {}");
}

#[test]
fn test_match_group_in_subtree() {
    let rules = create_rules(
        r#"
        macro_rules! foo {
            (fn $name:ident {$($i:ident)*} ) => ( fn $name() { $ (
                $ i ();
            )*} );
        }"#,
    );

    assert_expansion(MacroKind::Items, &rules, "foo! {fn baz {a b} }", "fn baz () {a () ; b () ;}");
}

#[test]
fn test_match_group_with_multichar_sep() {
    let rules = create_rules(
        r#"
        macro_rules! foo {
            (fn $name:ident {$($i:literal)*} ) => ( fn $name() -> bool { $($i)&&*} );
        }"#,
    );

    assert_expansion(
        MacroKind::Items,
        &rules,
        "foo! (fn baz {true true} );",
        "fn baz () -> bool {true &&true}",
    );
}

#[test]
fn test_match_group_zero_match() {
    let rules = create_rules(
        r#"
        macro_rules! foo {
            ( $($i:ident)* ) => ();
        }"#,
    );

    assert_expansion(MacroKind::Items, &rules, "foo! ();", "");
}

#[test]
fn test_match_group_in_group() {
    let rules = create_rules(
        r#"
        macro_rules! foo {
            { $( ( $($i:ident)* ) )* } => ( $( ( $($i)* ) )* );
        }"#,
    );

    assert_expansion(MacroKind::Items, &rules, "foo! ( (a b) );", "(a b)");
}

#[test]
fn test_expand_to_item_list() {
    let rules = create_rules(
        "
            macro_rules! structs {
                ($($i:ident),*) => {
                    $(struct $i { field: u32 } )*
                }
            }
            ",
    );
    let expansion = expand(&rules, "structs!(Foo, Bar);");
    let tree = token_tree_to_macro_items(&expansion).unwrap().tree();
    assert_eq!(
        format!("{:#?}", tree.syntax()).trim(),
        r#"
MACRO_ITEMS@[0; 40)
  STRUCT_DEF@[0; 20)
    STRUCT_KW@[0; 6) "struct"
    NAME@[6; 9)
      IDENT@[6; 9) "Foo"
    RECORD_FIELD_DEF_LIST@[9; 20)
      L_CURLY@[9; 10) "{"
      RECORD_FIELD_DEF@[10; 19)
        NAME@[10; 15)
          IDENT@[10; 15) "field"
        COLON@[15; 16) ":"
        PATH_TYPE@[16; 19)
          PATH@[16; 19)
            PATH_SEGMENT@[16; 19)
              NAME_REF@[16; 19)
                IDENT@[16; 19) "u32"
      R_CURLY@[19; 20) "}"
  STRUCT_DEF@[20; 40)
    STRUCT_KW@[20; 26) "struct"
    NAME@[26; 29)
      IDENT@[26; 29) "Bar"
    RECORD_FIELD_DEF_LIST@[29; 40)
      L_CURLY@[29; 30) "{"
      RECORD_FIELD_DEF@[30; 39)
        NAME@[30; 35)
          IDENT@[30; 35) "field"
        COLON@[35; 36) ":"
        PATH_TYPE@[36; 39)
          PATH@[36; 39)
            PATH_SEGMENT@[36; 39)
              NAME_REF@[36; 39)
                IDENT@[36; 39) "u32"
      R_CURLY@[39; 40) "}""#
            .trim()
    );
}

#[test]
fn test_expand_literals_to_token_tree() {
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

    let rules = create_rules(
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
    );
    let expansion = expand(&rules, "literals!(foo);");
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
fn test_two_idents() {
    let rules = create_rules(
        r#"
        macro_rules! foo {
            ($ i:ident, $ j:ident) => {
                fn foo() { let a = $ i; let b = $j; }
            }
        }
"#,
    );
    assert_expansion(
        MacroKind::Items,
        &rules,
        "foo! { foo, bar }",
        "fn foo () {let a = foo ; let b = bar ;}",
    );
}

#[test]
fn test_tt_to_stmts() {
    let rules = create_rules(
        r#"
        macro_rules! foo {
            () => {
                 let a = 0;
                 a = 10 + 1;
                 a
            }
        }
"#,
    );

    let expanded = expand(&rules, "foo!{}");
    let stmts = token_tree_to_macro_stmts(&expanded).unwrap().tree();

    assert_eq!(
        format!("{:#?}", stmts.syntax()).trim(),
        r#"MACRO_STMTS@[0; 15)
  LET_STMT@[0; 7)
    LET_KW@[0; 3) "let"
    BIND_PAT@[3; 4)
      NAME@[3; 4)
        IDENT@[3; 4) "a"
    EQ@[4; 5) "="
    LITERAL@[5; 6)
      INT_NUMBER@[5; 6) "0"
    SEMI@[6; 7) ";"
  EXPR_STMT@[7; 14)
    BIN_EXPR@[7; 13)
      PATH_EXPR@[7; 8)
        PATH@[7; 8)
          PATH_SEGMENT@[7; 8)
            NAME_REF@[7; 8)
              IDENT@[7; 8) "a"
      EQ@[8; 9) "="
      BIN_EXPR@[9; 13)
        LITERAL@[9; 11)
          INT_NUMBER@[9; 11) "10"
        PLUS@[11; 12) "+"
        LITERAL@[12; 13)
          INT_NUMBER@[12; 13) "1"
    SEMI@[13; 14) ";"
  EXPR_STMT@[14; 15)
    PATH_EXPR@[14; 15)
      PATH@[14; 15)
        PATH_SEGMENT@[14; 15)
          NAME_REF@[14; 15)
            IDENT@[14; 15) "a""#,
    );
}

#[test]
fn test_match_literal() {
    let rules = create_rules(
        r#"
    macro_rules! foo {
        ('(') => {
            fn foo() {}
        }
    }
"#,
    );
    assert_expansion(MacroKind::Items, &rules, "foo! ['('];", "fn foo () {}");
}

// The following tests are port from intellij-rust directly
// https://github.com/intellij-rust/intellij-rust/blob/c4e9feee4ad46e7953b1948c112533360b6087bb/src/test/kotlin/org/rust/lang/core/macros/RsMacroExpansionTest.kt

#[test]
fn test_path() {
    let rules = create_rules(
        r#"
        macro_rules! foo {
            ($ i:path) => {
                fn foo() { let a = $ i; }
            }
        }
"#,
    );
    assert_expansion(MacroKind::Items, &rules, "foo! { foo }", "fn foo () {let a = foo ;}");
    assert_expansion(
        MacroKind::Items,
        &rules,
        "foo! { bar::<u8>::baz::<u8> }",
        "fn foo () {let a = bar ::< u8 >:: baz ::< u8 > ;}",
    );
}

#[test]
fn test_two_paths() {
    let rules = create_rules(
        r#"
        macro_rules! foo {
            ($ i:path, $ j:path) => {
                fn foo() { let a = $ i; let b = $j; }
            }
        }
"#,
    );
    assert_expansion(
        MacroKind::Items,
        &rules,
        "foo! { foo, bar }",
        "fn foo () {let a = foo ; let b = bar ;}",
    );
}

#[test]
fn test_path_with_path() {
    let rules = create_rules(
        r#"
        macro_rules! foo {
            ($ i:path) => {
                fn foo() { let a = $ i :: bar; }
            }
        }
"#,
    );
    assert_expansion(MacroKind::Items, &rules, "foo! { foo }", "fn foo () {let a = foo :: bar ;}");
}

#[test]
fn test_expr() {
    let rules = create_rules(
        r#"
        macro_rules! foo {
            ($ i:expr) => {
                 fn bar() { $ i; }
            }
        }
"#,
    );

    assert_expansion(
        MacroKind::Items,
        &rules,
        "foo! { 2 + 2 * baz(3).quux() }",
        "fn bar () {2 + 2 * baz (3) . quux () ;}",
    );
}

#[test]
fn test_expr_order() {
    let rules = create_rules(
        r#"
        macro_rules! foo {
            ($ i:expr) => {
                 fn bar() { $ i * 2; }
            }
        }
"#,
    );
    let dump = format!("{:#?}", expand_to_items(&rules, "foo! { 1 + 1  }").syntax());
    assert_eq_text!(
        dump.trim(),
        r#"MACRO_ITEMS@[0; 15)
  FN_DEF@[0; 15)
    FN_KW@[0; 2) "fn"
    NAME@[2; 5)
      IDENT@[2; 5) "bar"
    PARAM_LIST@[5; 7)
      L_PAREN@[5; 6) "("
      R_PAREN@[6; 7) ")"
    BLOCK_EXPR@[7; 15)
      BLOCK@[7; 15)
        L_CURLY@[7; 8) "{"
        EXPR_STMT@[8; 14)
          BIN_EXPR@[8; 13)
            BIN_EXPR@[8; 11)
              LITERAL@[8; 9)
                INT_NUMBER@[8; 9) "1"
              PLUS@[9; 10) "+"
              LITERAL@[10; 11)
                INT_NUMBER@[10; 11) "1"
            STAR@[11; 12) "*"
            LITERAL@[12; 13)
              INT_NUMBER@[12; 13) "2"
          SEMI@[13; 14) ";"
        R_CURLY@[14; 15) "}""#,
    );
}

#[test]
fn test_last_expr() {
    let rules = create_rules(
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
    assert_expansion(
        MacroKind::Items,
        &rules,
        "vec!(1,2,3);",
        "{let mut v = Vec :: new () ; v . push (1) ; v . push (2) ; v . push (3) ; v}",
    );
}

#[test]
fn test_ty() {
    let rules = create_rules(
        r#"
        macro_rules! foo {
            ($ i:ty) => (
                fn bar() -> $ i { unimplemented!() }
            )
        }
"#,
    );
    assert_expansion(
        MacroKind::Items,
        &rules,
        "foo! { Baz<u8> }",
        "fn bar () -> Baz < u8 > {unimplemented ! ()}",
    );
}

#[test]
fn test_ty_with_complex_type() {
    let rules = create_rules(
        r#"
        macro_rules! foo {
            ($ i:ty) => (
                fn bar() -> $ i { unimplemented!() }
            )
        }
"#,
    );

    // Reference lifetime struct with generic type
    assert_expansion(
        MacroKind::Items,
        &rules,
        "foo! { &'a Baz<u8> }",
        "fn bar () -> & 'a Baz < u8 > {unimplemented ! ()}",
    );

    // extern "Rust" func type
    assert_expansion(
        MacroKind::Items,
        &rules,
        r#"foo! { extern "Rust" fn() -> Ret }"#,
        r#"fn bar () -> extern "Rust" fn () -> Ret {unimplemented ! ()}"#,
    );
}

#[test]
fn test_pat_() {
    let rules = create_rules(
        r#"
        macro_rules! foo {
            ($ i:pat) => { fn foo() { let $ i; } }
        }
"#,
    );
    assert_expansion(MacroKind::Items, &rules, "foo! { (a, b) }", "fn foo () {let (a , b) ;}");
}

#[test]
fn test_stmt() {
    let rules = create_rules(
        r#"
        macro_rules! foo {
            ($ i:stmt) => (
                fn bar() { $ i; }
            )
        }
"#,
    );
    assert_expansion(MacroKind::Items, &rules, "foo! { 2 }", "fn bar () {2 ;}");
    assert_expansion(MacroKind::Items, &rules, "foo! { let a = 0 }", "fn bar () {let a = 0 ;}");
}

#[test]
fn test_single_item() {
    let rules = create_rules(
        r#"
        macro_rules! foo {
            ($ i:item) => (
                $ i
            )
        }
"#,
    );
    assert_expansion(MacroKind::Items, &rules, "foo! {mod c {}}", "mod c {}");
}

#[test]
fn test_all_items() {
    let rules = create_rules(
        r#"
        macro_rules! foo {
            ($ ($ i:item)*) => ($ (
                $ i
            )*)
        }
"#,
    );
    assert_expansion(MacroKind::Items, &rules, r#"
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
"#, r#"extern crate a ; mod b ; mod c {} use d ; const E : i32 = 0 ; static F : i32 = 0 ; impl G {} struct H ; enum I {Foo} trait J {} fn h () {} extern {} type T = u8 ;"#);
}

#[test]
fn test_block() {
    let rules = create_rules(
        r#"
        macro_rules! foo {
            ($ i:block) => { fn foo() $ i }
        }
"#,
    );
    assert_expansion(MacroKind::Stmts, &rules, "foo! { { 1; } }", "fn foo () {1 ;}");
}

#[test]
fn test_meta() {
    let rules = create_rules(
        r#"
        macro_rules! foo {
            ($ i:meta) => (
                #[$ i]
                fn bar() {}
            )
        }
"#,
    );
    assert_expansion(
        MacroKind::Items,
        &rules,
        r#"foo! { cfg(target_os = "windows") }"#,
        r#"# [cfg (target_os = "windows")] fn bar () {}"#,
    );
}

#[test]
fn test_meta_doc_comments() {
    let rules = create_rules(
        r#"
        macro_rules! foo {
            ($(#[$ i:meta])+) => (
                $(#[$ i])+
                fn bar() {}
            )
        }
"#,
    );
    assert_expansion(
        MacroKind::Items,
        &rules,
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
fn test_tt_block() {
    let rules = create_rules(
        r#"
            macro_rules! foo {
                ($ i:tt) => { fn foo() $ i }
            }
    "#,
    );
    assert_expansion(MacroKind::Items, &rules, r#"foo! { { 1; } }"#, r#"fn foo () {1 ;}"#);
}

#[test]
fn test_tt_group() {
    let rules = create_rules(
        r#"
            macro_rules! foo {
                 ($($ i:tt)*) => { $($ i)* }
            }
    "#,
    );
    assert_expansion(MacroKind::Items, &rules, r#"foo! { fn foo() {} }"#, r#"fn foo () {}"#);
}
#[test]
fn test_lifetime() {
    let rules = create_rules(
        r#"
        macro_rules! foo {
              ($ lt:lifetime) => { struct Ref<$ lt>{ s: &$ lt str } }
        }
"#,
    );
    assert_expansion(MacroKind::Items, &rules, r#"foo!{'a}"#, r#"struct Ref <'a > {s : &'a str}"#);
}

#[test]
fn test_literal() {
    let rules = create_rules(
        r#"
        macro_rules! foo {
              ($ type:ty $ lit:literal) => { const VALUE: $ type = $ lit;};
        }
"#,
    );
    assert_expansion(MacroKind::Items, &rules, r#"foo!(u8 0);"#, r#"const VALUE : u8 = 0 ;"#);
}

#[test]
fn test_vis() {
    let rules = create_rules(
        r#"
        macro_rules! foo {
              ($ vis:vis $ name:ident) => { $ vis fn $ name() {}};
        }
"#,
    );
    assert_expansion(MacroKind::Items, &rules, r#"foo!(pub foo);"#, r#"pub fn foo () {}"#);

    // test optional casse
    assert_expansion(MacroKind::Items, &rules, r#"foo!(foo);"#, r#"fn foo () {}"#);
}

#[test]
fn test_inner_macro_rules() {
    let rules = create_rules(
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
    );
    assert_expansion(
        MacroKind::Items,
        &rules,
        r#"foo!(x,y, 1);"#,
        r#"macro_rules ! bar {($ bi : ident) => {fn $ bi () -> u8 {1}}} bar ! (x) ; fn y () -> u8 {1}"#,
    );
}

// The following tests are based on real world situations
#[test]
fn test_vec() {
    let rules = create_rules(
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
    assert_expansion(MacroKind::Items, &rules, r#"vec!();"#, r#"{let mut v = Vec :: new () ; v}"#);
    assert_expansion(
        MacroKind::Items,
        &rules,
        r#"vec![1u32,2];"#,
        r#"{let mut v = Vec :: new () ; v . push (1u32) ; v . push (2) ; v}"#,
    );

    assert_eq!(
        format!("{:#?}", expand_to_expr(&rules, r#"vec![1u32,2];"#).syntax()).trim(),
        r#"BLOCK_EXPR@[0; 45)
  BLOCK@[0; 45)
    L_CURLY@[0; 1) "{"
    LET_STMT@[1; 20)
      LET_KW@[1; 4) "let"
      BIND_PAT@[4; 8)
        MUT_KW@[4; 7) "mut"
        NAME@[7; 8)
          IDENT@[7; 8) "v"
      EQ@[8; 9) "="
      CALL_EXPR@[9; 19)
        PATH_EXPR@[9; 17)
          PATH@[9; 17)
            PATH@[9; 12)
              PATH_SEGMENT@[9; 12)
                NAME_REF@[9; 12)
                  IDENT@[9; 12) "Vec"
            COLONCOLON@[12; 14) "::"
            PATH_SEGMENT@[14; 17)
              NAME_REF@[14; 17)
                IDENT@[14; 17) "new"
        ARG_LIST@[17; 19)
          L_PAREN@[17; 18) "("
          R_PAREN@[18; 19) ")"
      SEMI@[19; 20) ";"
    EXPR_STMT@[20; 33)
      METHOD_CALL_EXPR@[20; 32)
        PATH_EXPR@[20; 21)
          PATH@[20; 21)
            PATH_SEGMENT@[20; 21)
              NAME_REF@[20; 21)
                IDENT@[20; 21) "v"
        DOT@[21; 22) "."
        NAME_REF@[22; 26)
          IDENT@[22; 26) "push"
        ARG_LIST@[26; 32)
          L_PAREN@[26; 27) "("
          LITERAL@[27; 31)
            INT_NUMBER@[27; 31) "1u32"
          R_PAREN@[31; 32) ")"
      SEMI@[32; 33) ";"
    EXPR_STMT@[33; 43)
      METHOD_CALL_EXPR@[33; 42)
        PATH_EXPR@[33; 34)
          PATH@[33; 34)
            PATH_SEGMENT@[33; 34)
              NAME_REF@[33; 34)
                IDENT@[33; 34) "v"
        DOT@[34; 35) "."
        NAME_REF@[35; 39)
          IDENT@[35; 39) "push"
        ARG_LIST@[39; 42)
          L_PAREN@[39; 40) "("
          LITERAL@[40; 41)
            INT_NUMBER@[40; 41) "2"
          R_PAREN@[41; 42) ")"
      SEMI@[42; 43) ";"
    PATH_EXPR@[43; 44)
      PATH@[43; 44)
        PATH_SEGMENT@[43; 44)
          NAME_REF@[43; 44)
            IDENT@[43; 44) "v"
    R_CURLY@[44; 45) "}""#
    );
}

#[test]
fn test_winapi_struct() {
    // from https://github.com/retep998/winapi-rs/blob/a7ef2bca086aae76cf6c4ce4c2552988ed9798ad/src/macros.rs#L366

    let rules = create_rules(
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
    );
    // from https://github.com/retep998/winapi-rs/blob/a7ef2bca086aae76cf6c4ce4c2552988ed9798ad/src/shared/d3d9caps.rs
    assert_expansion(MacroKind::Items, &rules, r#"STRUCT!{struct D3DVSHADERCAPS2_0 {Caps: u8,}}"#,
        "# [repr (C)] # [derive (Copy)] pub struct D3DVSHADERCAPS2_0 {pub Caps : u8 ,} impl Clone for D3DVSHADERCAPS2_0 {# [inline] fn clone (& self) -> D3DVSHADERCAPS2_0 {* self}} # [cfg (feature = \"impl-default\")] impl Default for D3DVSHADERCAPS2_0 {# [inline] fn default () -> D3DVSHADERCAPS2_0 {unsafe {$crate :: _core :: mem :: zeroed ()}}}");
    assert_expansion(MacroKind::Items, &rules, r#"STRUCT!{#[cfg_attr(target_arch = "x86", repr(packed))] struct D3DCONTENTPROTECTIONCAPS {Caps : u8 ,}}"#,
        "# [repr (C)] # [derive (Copy)] # [cfg_attr (target_arch = \"x86\" , repr (packed))] pub struct D3DCONTENTPROTECTIONCAPS {pub Caps : u8 ,} impl Clone for D3DCONTENTPROTECTIONCAPS {# [inline] fn clone (& self) -> D3DCONTENTPROTECTIONCAPS {* self}} # [cfg (feature = \"impl-default\")] impl Default for D3DCONTENTPROTECTIONCAPS {# [inline] fn default () -> D3DCONTENTPROTECTIONCAPS {unsafe {$crate :: _core :: mem :: zeroed ()}}}");
}

#[test]
fn test_int_base() {
    let rules = create_rules(
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
    );

    assert_expansion(MacroKind::Items, &rules, r#" int_base!{Binary for isize as usize -> Binary}"#,
        "# [stable (feature = \"rust1\" , since = \"1.0.0\")] impl fmt ::Binary for isize {fn fmt (& self , f : & mut fmt :: Formatter < \'_ >) -> fmt :: Result {Binary . fmt_int (* self as usize , f)}}"
        );
}

#[test]
fn test_generate_pattern_iterators() {
    // from https://github.com/rust-lang/rust/blob/316a391dcb7d66dc25f1f9a4ec9d368ef7615005/src/libcore/str/mod.rs
    let rules = create_rules(
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
    );

    assert_expansion(MacroKind::Items, &rules, r#"generate_pattern_iterators ! ( double ended ; with # [ stable ( feature = "rust1" , since = "1.0.0" ) ] , Split , RSplit , & 'a str );"#,
        "fn foo () {}");
}

#[test]
fn test_impl_fn_for_zst() {
    // from https://github.com/rust-lang/rust/blob/5d20ff4d2718c820632b38c1e49d4de648a9810b/src/libcore/internal_macros.rs
    let rules = create_rules(
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
    );

    assert_expansion(MacroKind::Items, &rules, r#"
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
        "# [derive (Clone)] struct CharEscapeDebugContinue ; impl Fn < (char ,) > for CharEscapeDebugContinue {# [inline] extern \"rust-call\" fn call (& self , (c ,) : (char ,)) -> char :: EscapeDebug {{c . escape_debug_ext (false)}}} impl FnMut < (char ,) > for CharEscapeDebugContinue {# [inline] extern \"rust-call\" fn call_mut (& mut self , (c ,) : (char ,)) -> char :: EscapeDebug {Fn :: call (&* self , (c ,))}} impl FnOnce < (char ,) > for CharEscapeDebugContinue {type Output = char :: EscapeDebug ; # [inline] extern \"rust-call\" fn call_once (self , (c ,) : (char ,)) -> char :: EscapeDebug {Fn :: call (& self , (c ,))}} # [derive (Clone)] struct CharEscapeUnicode ; impl Fn < (char ,) > for CharEscapeUnicode {# [inline] extern \"rust-call\" fn call (& self , (c ,) : (char ,)) -> char :: EscapeUnicode {{c . escape_unicode ()}}} impl FnMut < (char ,) > for CharEscapeUnicode {# [inline] extern \"rust-call\" fn call_mut (& mut self , (c ,) : (char ,)) -> char :: EscapeUnicode {Fn :: call (&* self , (c ,))}} impl FnOnce < (char ,) > for CharEscapeUnicode {type Output = char :: EscapeUnicode ; # [inline] extern \"rust-call\" fn call_once (self , (c ,) : (char ,)) -> char :: EscapeUnicode {Fn :: call (& self , (c ,))}} # [derive (Clone)] struct CharEscapeDefault ; impl Fn < (char ,) > for CharEscapeDefault {# [inline] extern \"rust-call\" fn call (& self , (c ,) : (char ,)) -> char :: EscapeDefault {{c . escape_default ()}}} impl FnMut < (char ,) > for CharEscapeDefault {# [inline] extern \"rust-call\" fn call_mut (& mut self , (c ,) : (char ,)) -> char :: EscapeDefault {Fn :: call (&* self , (c ,))}} impl FnOnce < (char ,) > for CharEscapeDefault {type Output = char :: EscapeDefault ; # [inline] extern \"rust-call\" fn call_once (self , (c ,) : (char ,)) -> char :: EscapeDefault {Fn :: call (& self , (c ,))}}");
}

#[test]
fn test_impl_nonzero_fmt() {
    // from https://github.com/rust-lang/rust/blob/316a391dcb7d66dc25f1f9a4ec9d368ef7615005/src/libcore/num/mod.rs#L12
    let rules = create_rules(
        r#"
        macro_rules! impl_nonzero_fmt {
            ( #[$stability: meta] ( $( $Trait: ident ),+ ) for $Ty: ident ) => {
                fn foo () {}
            }
        }
"#,
    );

    assert_expansion(MacroKind::Items, &rules, r#"impl_nonzero_fmt! { # [stable(feature= "nonzero",since="1.28.0")] (Debug,Display,Binary,Octal,LowerHex,UpperHex) for NonZeroU8}"#,
        "fn foo () {}");
}

#[test]
fn test_cfg_if_items() {
    // from https://github.com/rust-lang/rust/blob/33fe1131cadba69d317156847be9a402b89f11bb/src/libstd/macros.rs#L986
    let rules = create_rules(
        r#"
        macro_rules! __cfg_if_items {
            (($($not:meta,)*) ; ) => {};
            (($($not:meta,)*) ; ( ($($m:meta),*) ($($it:item)*) ), $($rest:tt)*) => {
                 __cfg_if_items! { ($($not,)* $($m,)*) ; $($rest)* }
            }
        }
"#,
    );

    assert_expansion(MacroKind::Items, &rules, r#"__cfg_if_items ! { ( rustdoc , ) ; ( ( ) ( # [ cfg ( any ( target_os = "redox" , unix ) ) ] # [ stable ( feature = "rust1" , since = "1.0.0" ) ] pub use sys :: ext as unix ; # [ cfg ( windows ) ] # [ stable ( feature = "rust1" , since = "1.0.0" ) ] pub use sys :: ext as windows ; # [ cfg ( any ( target_os = "linux" , target_os = "l4re" ) ) ] pub mod linux ; ) ) , }"#,
        "__cfg_if_items ! {(rustdoc ,) ;}");
}

#[test]
fn test_cfg_if_main() {
    // from https://github.com/rust-lang/rust/blob/3d211248393686e0f73851fc7548f6605220fbe1/src/libpanic_unwind/macros.rs#L9
    let rules = create_rules(
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
    );

    assert_expansion(MacroKind::Items, &rules, r#"
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
        "__cfg_if_items ! {() ; ((target_env = \"msvc\") ()) , ((all (target_arch = \"wasm32\" , not (target_os = \"emscripten\"))) ()) , (() (mod libunwind ; pub use libunwind :: * ;)) ,}");

    assert_expansion(MacroKind::Items, &rules, r#"
cfg_if ! { @ __apply cfg ( all ( not ( any ( not ( any ( target_os = "solaris" , target_os = "illumos" ) ) ) ) ) ) , }
"#,
        ""
    );
}

#[test]
fn test_proptest_arbitrary() {
    // from https://github.com/AltSysrq/proptest/blob/d1c4b049337d2f75dd6f49a095115f7c532e5129/proptest/src/arbitrary/macros.rs#L16
    let rules = create_rules(
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
    );

    assert_expansion(MacroKind::Items, &rules, r#"arbitrary !   ( [ A : Arbitrary ]
        Vec < A > ,
        VecStrategy < A :: Strategy > ,
        RangedParams1 < A :: Parameters > ;
        args =>   { let product_unpack !   [ range , a ] = args ; vec ( any_with :: < A >   ( a ) , range ) }
    ) ;"#,
    "impl <A : Arbitrary > $crate :: arbitrary :: Arbitrary for Vec < A > {type Parameters = RangedParams1 < A :: Parameters > ; type Strategy = VecStrategy < A :: Strategy > ; fn arbitrary_with (args : Self :: Parameters) -> Self :: Strategy {{let product_unpack ! [range , a] = args ; vec (any_with :: < A > (a) , range)}}}");
}

#[test]
fn test_old_ridl() {
    // This is from winapi 2.8, which do not have a link from github
    //
    let rules = create_rules(
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
    );

    let expanded = expand(&rules, r#"
RIDL!{interface ID3D11Asynchronous(ID3D11AsynchronousVtbl): ID3D11DeviceChild(ID3D11DeviceChildVtbl) {
    fn GetDataSize(&mut self) -> UINT
}}"#);
    assert_eq!(expanded.to_string(), "impl ID3D11Asynchronous {pub unsafe fn GetDataSize (& mut self) -> UINT {((* self . lpVtbl) .GetDataSize) (self)}}");
}

#[test]
fn test_quick_error() {
    let rules = create_rules(
        r#"
macro_rules! quick_error {

 (SORT [enum $name:ident $( #[$meta:meta] )*]
        items [$($( #[$imeta:meta] )*
                  => $iitem:ident: $imode:tt [$( $ivar:ident: $ityp:ty ),*]
                                {$( $ifuncs:tt )*} )* ]
        buf [ ]
        queue [ ]
    ) => {
        quick_error!(ENUM_DEFINITION [enum $name $( #[$meta] )*]
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
    );

    let expanded = expand(
        &rules,
        r#"
quick_error ! (SORT [enum Wrapped # [derive (Debug)]] items [
        => One : UNIT [] {}
        => Two : TUPLE [s :String] {display ("two: {}" , s) from ()}
    ] buf [] queue []) ;
"#,
    );

    assert_eq!(expanded.to_string(), "quick_error ! (ENUM_DEFINITION [enum Wrapped # [derive (Debug)]] body [] queue [=> One : UNIT [] => Two : TUPLE [s : String]]) ;");
}

#[test]
fn test_empty_repeat_vars_in_empty_repeat_vars() {
    let rules = create_rules(r#"
macro_rules! delegate_impl {
    ([$self_type:ident, $self_wrap:ty, $self_map:ident]
     pub trait $name:ident $(: $sup:ident)* $(+ $more_sup:ident)* {

        // "Escaped" associated types. Stripped before making the `trait`
        // itself, but forwarded when delegating impls.
        $(
        @escape [type $assoc_name_ext:ident]
        // Associated types. Forwarded.
        )*
        $(
        @section type
        $(
            $(#[$_assoc_attr:meta])*
            type $assoc_name:ident $(: $assoc_bound:ty)*;
        )+
        )*
        // Methods. Forwarded. Using $self_map!(self) around the self argument.
        // Methods must use receiver `self` or explicit type like `self: &Self`
        // &self and &mut self are _not_ supported.
        $(
        @section self
        $(
            $(#[$_method_attr:meta])*
            fn $method_name:ident(self $(: $self_selftype:ty)* $(,$marg:ident : $marg_ty:ty)*) -> $mret:ty;
        )+
        )*
        // Arbitrary tail that is ignored when forwarding.
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
"#);

    assert_expansion(MacroKind::Items, &rules, r#"delegate_impl ! {[G , & 'a mut G , deref] pub trait Data : GraphBase {@ section type type NodeWeight ;}}"#, "impl <> Data for & \'a mut G where G : Data {}");
}
