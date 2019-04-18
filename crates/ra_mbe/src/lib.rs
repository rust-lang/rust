/// `mbe` (short for Macro By Example) crate contains code for handling
/// `macro_rules` macros. It uses `TokenTree` (from `ra_tt` package) as the
/// interface, although it contains some code to bridge `SyntaxNode`s and
/// `TokenTree`s as well!

macro_rules! impl_froms {
    ($e:ident: $($v:ident), *) => {
        $(
            impl From<$v> for $e {
                fn from(it: $v) -> $e {
                    $e::$v(it)
                }
            }
        )*
    }
}

// mod tt_cursor;
mod mbe_parser;
mod mbe_expander;
mod syntax_bridge;
mod tt_cursor;
mod subtree_source;
mod subtree_parser;

use ra_syntax::SmolStr;

pub use tt::{Delimiter, Punct};

#[derive(Debug, PartialEq, Eq)]
pub enum ParseError {
    Expected(String),
}

#[derive(Debug, PartialEq, Eq)]
pub enum ExpandError {
    NoMatchingRule,
    UnexpectedToken,
    BindingError(String),
    ConversionError,
}

pub use crate::syntax_bridge::{
    ast_to_token_tree,
    token_tree_to_ast_item_list,
    syntax_node_to_token_tree,
    token_tree_to_macro_items,
};

/// This struct contains AST for a single `macro_rules` definition. What might
/// be very confusing is that AST has almost exactly the same shape as
/// `tt::TokenTree`, but there's a crucial difference: in macro rules, `$ident`
/// and `$()*` have special meaning (see `Var` and `Repeat` data structures)
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MacroRules {
    pub(crate) rules: Vec<Rule>,
}

impl MacroRules {
    pub fn parse(tt: &tt::Subtree) -> Result<MacroRules, ParseError> {
        mbe_parser::parse(tt)
    }
    pub fn expand(&self, tt: &tt::Subtree) -> Result<tt::Subtree, ExpandError> {
        mbe_expander::expand(self, tt)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Rule {
    pub(crate) lhs: Subtree,
    pub(crate) rhs: Subtree,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum TokenTree {
    Leaf(Leaf),
    Subtree(Subtree),
    Repeat(Repeat),
}
impl_froms!(TokenTree: Leaf, Subtree, Repeat);

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Leaf {
    Literal(Literal),
    Punct(Punct),
    Ident(Ident),
    Var(Var),
}
impl_froms!(Leaf: Literal, Punct, Ident, Var);

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Subtree {
    pub(crate) delimiter: Delimiter,
    pub(crate) token_trees: Vec<TokenTree>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Repeat {
    pub(crate) subtree: Subtree,
    pub(crate) kind: RepeatKind,
    pub(crate) separator: Option<char>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum RepeatKind {
    ZeroOrMore,
    OneOrMore,
    ZeroOrOne,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Literal {
    pub(crate) text: SmolStr,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Ident {
    pub(crate) text: SmolStr,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Var {
    pub(crate) text: SmolStr,
    pub(crate) kind: Option<SmolStr>,
}

#[cfg(test)]
mod tests {
    use ra_syntax::{ast, AstNode};

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

        let source_file = ast::SourceFile::parse(macro_definition);
        let macro_definition =
            source_file.syntax().descendants().find_map(ast::MacroCall::cast).unwrap();

        let source_file = ast::SourceFile::parse(macro_invocation);
        let macro_invocation =
            source_file.syntax().descendants().find_map(ast::MacroCall::cast).unwrap();

        let (definition_tt, _) = ast_to_token_tree(macro_definition.token_tree().unwrap()).unwrap();
        let (invocation_tt, _) = ast_to_token_tree(macro_invocation.token_tree().unwrap()).unwrap();
        let rules = crate::MacroRules::parse(&definition_tt).unwrap();
        let expansion = rules.expand(&invocation_tt).unwrap();
        assert_eq!(
        expansion.to_string(),
        "impl From < Leaf > for TokenTree {fn from (it : Leaf) -> TokenTree {TokenTree :: Leaf (it)}} \
         impl From < Subtree > for TokenTree {fn from (it : Subtree) -> TokenTree {TokenTree :: Subtree (it)}}"
    )
    }

    pub(crate) fn create_rules(macro_definition: &str) -> MacroRules {
        let source_file = ast::SourceFile::parse(macro_definition);
        let macro_definition =
            source_file.syntax().descendants().find_map(ast::MacroCall::cast).unwrap();

        let (definition_tt, _) = ast_to_token_tree(macro_definition.token_tree().unwrap()).unwrap();
        crate::MacroRules::parse(&definition_tt).unwrap()
    }

    pub(crate) fn expand(rules: &MacroRules, invocation: &str) -> tt::Subtree {
        let source_file = ast::SourceFile::parse(invocation);
        let macro_invocation =
            source_file.syntax().descendants().find_map(ast::MacroCall::cast).unwrap();

        let (invocation_tt, _) = ast_to_token_tree(macro_invocation.token_tree().unwrap()).unwrap();

        rules.expand(&invocation_tt).unwrap()
    }

    pub(crate) fn expand_to_syntax(
        rules: &MacroRules,
        invocation: &str,
    ) -> ra_syntax::TreeArc<ast::MacroItems> {
        let expanded = expand(rules, invocation);
        token_tree_to_macro_items(&expanded)
    }

    pub(crate) fn assert_expansion(rules: &MacroRules, invocation: &str, expansion: &str) {
        let expanded = expand(rules, invocation);
        assert_eq!(expanded.to_string(), expansion);

        let tree = token_tree_to_macro_items(&expanded);

        // Eat all white space by parse it back and forth
        let expansion = ast::SourceFile::parse(expansion);
        let expansion = syntax_node_to_token_tree(expansion.syntax()).unwrap().0;
        let file = token_tree_to_macro_items(&expansion);

        assert_eq!(tree.syntax().debug_dump().trim(), file.syntax().debug_dump().trim());
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

        assert_expansion(&rules, "foo! { foo }", "mod foo {}");
        assert_expansion(&rules, "foo! { = bar }", "fn bar () {}");
        assert_expansion(&rules, "foo! { + Baz }", "struct Baz ;");
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

        assert_expansion(&rules, "foo! { foo }", "mod foo {}");
        assert_expansion(&rules, "foo! { bar = }", "fn bar () {}");
        assert_expansion(&rules, "foo! { Baz + }", "struct Baz ;");
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

        assert_expansion(&rules, "foo! { foo }", "mod foo {}");
        assert_expansion(&rules, "foo! { spam bar }", "fn bar () {}");
        assert_expansion(&rules, "foo! { eggs Baz }", "struct Baz ;");
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

        assert_expansion(&rules, "foo! { foo, bar }", "mod foo {} mod bar {}");
        assert_expansion(&rules, "foo! { foo# bar }", "fn foo () {} fn bar () {}");
        assert_expansion(&rules, "foo! { Foo,# Bar }", "struct Foo ; struct Bar ;");
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

        assert_expansion(&rules, "foo! { foo, bar }", "struct Bar {fn foo {} fn bar {}}");
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

        assert_expansion(&rules, "foo! { foo, bar }", "fn baz {foo () ; bar () ;}");
    }

    #[test]
    fn expand_to_item_list() {
        let rules = create_rules(
            "
            macro_rules! structs {
                ($($i:ident),*) => {
                    $(struct $i { field: u32 } )*
                }
            }
            ",
        );
        let expansion = expand(&rules, "structs!(Foo, Bar)");
        let tree = token_tree_to_macro_items(&expansion);
        assert_eq!(
            tree.syntax().debug_dump().trim(),
            r#"
MACRO_ITEMS@[0; 40)
  STRUCT_DEF@[0; 20)
    STRUCT_KW@[0; 6) "struct"
    NAME@[6; 9)
      IDENT@[6; 9) "Foo"
    NAMED_FIELD_DEF_LIST@[9; 20)
      L_CURLY@[9; 10) "{"
      NAMED_FIELD_DEF@[10; 19)
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
    NAMED_FIELD_DEF_LIST@[29; 40)
      L_CURLY@[29; 30) "{"
      NAMED_FIELD_DEF@[30; 39)
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
    fn expand_literals_to_token_tree() {
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
        let expansion = expand(&rules, "literals!(foo)");
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
        assert_expansion(&rules, "foo! { foo, bar }", "fn foo () {let a = foo ; let b = bar ;}");
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
        assert_expansion(&rules, "foo! { foo }", "fn foo () {let a = foo ;}");
        assert_expansion(
            &rules,
            "foo! { bar::<u8>::baz::<u8> }",
            "fn foo () {let a = bar :: < u8 > :: baz :: < u8 > ;}",
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
        assert_expansion(&rules, "foo! { foo, bar }", "fn foo () {let a = foo ; let b = bar ;}");
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
        assert_expansion(&rules, "foo! { foo }", "fn foo () {let a = foo :: bar ;}");
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

        assert_eq!(
            expand_to_syntax(&rules, "foo! { 1 + 1  }").syntax().debug_dump().trim(),
            r#"MACRO_ITEMS@[0; 15)
  FN_DEF@[0; 15)
    FN_KW@[0; 2) "fn"
    NAME@[2; 5)
      IDENT@[2; 5) "bar"
    PARAM_LIST@[5; 7)
      L_PAREN@[5; 6) "("
      R_PAREN@[6; 7) ")"
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
            &rules,
            "foo! { Baz<u8> }",
            "fn bar () -> Baz < u8 > {unimplemented ! ()}",
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
        assert_expansion(&rules, "foo! { (a, b) }", "fn foo () {let (a , b) ;}");
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
        assert_expansion(&rules, "foo! { 2 }", "fn bar () {2 ;}");
        assert_expansion(&rules, "foo! { let a = 0 }", "fn bar () {let a = 0 ;}");
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
        assert_expansion(&rules, "foo! {mod c {}}", "mod c {}");
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
        assert_expansion(&rules, r#"
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
}
