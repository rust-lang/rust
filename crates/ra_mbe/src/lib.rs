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
use smallvec::SmallVec;

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
    token_tree_to_expr,
    token_tree_to_pat,
    token_tree_to_ty,
    token_tree_to_macro_items,
    token_tree_to_macro_stmts,
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

#[derive(Clone, Debug, Eq)]
pub(crate) enum Separator {
    Literal(tt::Literal),
    Ident(tt::Ident),
    Puncts(SmallVec<[tt::Punct; 3]>),
}

// Note that when we compare a Separator, we just care about its textual value.
impl PartialEq for crate::Separator {
    fn eq(&self, other: &crate::Separator) -> bool {
        use crate::Separator::*;

        match (self, other) {
            (Ident(ref a), Ident(ref b)) => a.text == b.text,
            (Literal(ref a), Literal(ref b)) => a.text == b.text,
            (Puncts(ref a), Puncts(ref b)) if a.len() == b.len() => {
                let a_iter = a.iter().map(|a| a.char);
                let b_iter = b.iter().map(|b| b.char);
                a_iter.eq(b_iter)
            }
            _ => false,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Repeat {
    pub(crate) subtree: Subtree,
    pub(crate) kind: RepeatKind,
    pub(crate) separator: Option<Separator>,
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
        "impl From <Leaf > for TokenTree {fn from (it : Leaf) -> TokenTree {TokenTree ::Leaf (it)}} \
         impl From <Subtree > for TokenTree {fn from (it : Subtree) -> TokenTree {TokenTree ::Subtree (it)}}"
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

    pub(crate) fn expand_to_items(
        rules: &MacroRules,
        invocation: &str,
    ) -> ra_syntax::TreeArc<ast::MacroItems> {
        let expanded = expand(rules, invocation);
        token_tree_to_macro_items(&expanded).unwrap()
    }

    #[allow(unused)]
    pub(crate) fn expand_to_stmts(
        rules: &MacroRules,
        invocation: &str,
    ) -> ra_syntax::TreeArc<ast::MacroStmts> {
        let expanded = expand(rules, invocation);
        token_tree_to_macro_stmts(&expanded).unwrap()
    }

    pub(crate) fn expand_to_expr(
        rules: &MacroRules,
        invocation: &str,
    ) -> ra_syntax::TreeArc<ast::Expr> {
        let expanded = expand(rules, invocation);
        token_tree_to_expr(&expanded).unwrap()
    }

    pub(crate) fn assert_expansion(
        rules: &MacroRules,
        invocation: &str,
        expansion: &str,
    ) -> tt::Subtree {
        let expanded = expand(rules, invocation);
        assert_eq!(expanded.to_string(), expansion);

        // FIXME: Temp comment below code
        // It is because after the lexer change,
        // The SyntaxNode structure cannot be matched easily

        // let tree = token_tree_to_macro_items(&expanded);

        // // Eat all white space by parse it back and forth
        // // Because $crate will seperate in two token , will do some special treatment here
        // let expansion = expansion.replace("$crate", "C_C__C");
        // let expansion = ast::SourceFile::parse(&expansion);
        // let expansion = syntax_node_to_token_tree(expansion.syntax()).unwrap().0;
        // let file = token_tree_to_macro_items(&expansion);
        // let file = file.unwrap().syntax().debug_dump().trim().to_string();
        // let tree = tree.unwrap().syntax().debug_dump().trim().to_string();

        // let file = file.replace("C_C__C", "$crate");
        // assert_eq!(tree, file,);

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

        assert_expansion(&rules, "foo! { foo, bar }", "fn baz {foo () ;bar ()}");
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

        assert_expansion(&rules, "foo! {#abc}", "fn baz {}");
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

        assert_expansion(&rules, "foo! {fn baz {a b} }", "fn baz () {a () ; b () ;}");
    }

    #[test]
    fn test_match_group_with_multichar_sep() {
        let rules = create_rules(
            r#"
        macro_rules! foo {            
            (fn $name:ident {$($i:literal)*} ) => ( fn $name() -> bool { $($i)&&*} );            
        }"#,
        );

        assert_expansion(&rules, "foo! (fn baz {true true} )", "fn baz () -> bool {true &&true}");
    }

    #[test]
    fn test_match_group_zero_match() {
        let rules = create_rules(
            r#"
        macro_rules! foo {            
            ( $($i:ident)* ) => ();            
        }"#,
        );

        assert_expansion(&rules, "foo! ()", "");
    }

    #[test]
    fn test_match_group_in_group() {
        let rules = create_rules(
            r#"
        macro_rules! foo {            
            { $( ( $($i:ident)* ) )* } => ( $( ( $($i)* ) )* );
        }"#,
        );

        assert_expansion(&rules, "foo! ( (a b) )", "(a b)");
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
        let expansion = expand(&rules, "structs!(Foo, Bar)");
        let tree = token_tree_to_macro_items(&expansion);
        assert_eq!(
            tree.unwrap().syntax().debug_dump().trim(),
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
        let stmts = token_tree_to_macro_stmts(&expanded);

        assert_eq!(
            stmts.unwrap().syntax().debug_dump().trim(),
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
            expand_to_items(&rules, "foo! { 1 + 1  }").syntax().debug_dump().trim(),
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
            &rules,
            "vec!(1,2,3)",
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
            &rules,
            "foo! { &'a Baz<u8> }",
            "fn bar () -> & 'a Baz < u8 > {unimplemented ! ()}",
        );

        // extern "Rust" func type
        assert_expansion(
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

    #[test]
    fn test_block() {
        let rules = create_rules(
            r#"
        macro_rules! foo {
            ($ i:block) => { fn foo() $ i }
        }
"#,
        );
        assert_expansion(&rules, "foo! { { 1; } }", "fn foo () {1 ;}");
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
            &rules,
            r#"foo! { cfg(target_os = "windows") }"#,
            r#"# [cfg (target_os = "windows")] fn bar () {}"#,
        );
    }

    #[test]
    // fn test_tt_block() {
    //     let rules = create_rules(
    //         r#"
    //         macro_rules! foo {
    //             ($ i:tt) => { fn foo() $ i }
    //         }
    // "#,
    //     );
    //     assert_expansion(&rules, r#"foo! { { 1; } }"#, r#"fn foo () {1 ;}"#);
    // }

    // #[test]
    // fn test_tt_group() {
    //     let rules = create_rules(
    //         r#"
    //         macro_rules! foo {
    //              ($($ i:tt)*) => { $($ i)* }
    //         }
    // "#,
    //     );
    //     assert_expansion(&rules, r#"foo! { fn foo() {} }"#, r#"fn foo () {}"#);
    // }
    #[test]
    fn test_lifetime() {
        let rules = create_rules(
            r#"
        macro_rules! foo {
              ($ lt:lifetime) => { struct Ref<$ lt>{ s: &$ lt str } }
        }
"#,
        );
        assert_expansion(&rules, r#"foo!{'a}"#, r#"struct Ref <'a > {s : &'a str}"#);
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
        assert_expansion(&rules, r#"foo!(u8 0)"#, r#"const VALUE : u8 = 0 ;"#);
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
        assert_expansion(&rules, r#"foo!(pub foo);"#, r#"pub fn foo () {}"#);
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
        assert_expansion(&rules, r#"vec!();"#, r#"{let mut v = Vec :: new () ;  v}"#);
        assert_expansion(
            &rules,
            r#"vec![1u32,2]"#,
            r#"{let mut v = Vec :: new () ; v . push (1u32) ; v . push (2) ; v}"#,
        );

        assert_eq!(
            expand_to_expr(&rules, r#"vec![1u32,2]"#).syntax().debug_dump().trim(),
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
        assert_expansion(&rules, r#"STRUCT!{struct D3DVSHADERCAPS2_0 {Caps: u8,}}"#,
        "# [repr (C)] # [derive (Copy)]  pub struct D3DVSHADERCAPS2_0 {pub Caps : u8 ,} impl Clone for D3DVSHADERCAPS2_0 {# [inline] fn clone (& self) -> D3DVSHADERCAPS2_0 {* self}} # [cfg (feature = \"impl-default\")] impl Default for D3DVSHADERCAPS2_0 {# [inline] fn default () -> D3DVSHADERCAPS2_0 {unsafe {$crate :: _core :: mem :: zeroed ()}}}");
        assert_expansion(&rules, r#"STRUCT!{#[cfg_attr(target_arch = "x86", repr(packed))] struct D3DCONTENTPROTECTIONCAPS {Caps : u8 ,}}"#, 
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

        assert_expansion(&rules, r#" int_base!{Binary for isize as usize -> Binary}"#, 
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

        assert_expansion(&rules, r#"generate_pattern_iterators ! ( double ended ; with # [ stable ( feature = "rust1" , since = "1.0.0" ) ] , Split , RSplit , & 'a str )"#, 
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
}
"#,
        );

        assert_expansion(&rules, r#"
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
        "# [derive (Clone)] struct CharEscapeDebugContinue ; impl  Fn < (char ,) > for CharEscapeDebugContinue {# [inline] extern \"rust-call\" fn call (& self , (c ,) : (char ,)) -> char :: EscapeDebug {{c . escape_debug_ext (false)}}} impl  FnMut < (char ,) > for CharEscapeDebugContinue {# [inline] extern \"rust-call\" fn call_mut (& mut self , (c ,) : (char ,)) -> char :: EscapeDebug {Fn :: call (&* self , (c ,))}} impl  FnOnce < (char ,) > for CharEscapeDebugContinue {type Output = char :: EscapeDebug ; # [inline] extern \"rust-call\" fn call_once (self , (c ,) : (char ,)) -> char :: EscapeDebug {Fn :: call (& self , (c ,))}} # [derive (Clone)] struct CharEscapeUnicode ; impl  Fn < (char ,) > for CharEscapeUnicode {# [inline] extern \"rust-call\" fn call (& self , (c ,) : (char ,)) -> char :: EscapeUnicode {{c . escape_unicode ()}}} impl  FnMut < (char ,) > for CharEscapeUnicode {# [inline] extern \"rust-call\" fn call_mut (& mut self , (c ,) : (char ,)) -> char :: EscapeUnicode {Fn :: call (&* self , (c ,))}} impl  FnOnce < (char ,) > for CharEscapeUnicode {type Output = char :: EscapeUnicode ; # [inline] extern \"rust-call\" fn call_once (self , (c ,) : (char ,)) -> char :: EscapeUnicode {Fn :: call (& self , (c ,))}} # [derive (Clone)] struct CharEscapeDefault ; impl  Fn < (char ,) > for CharEscapeDefault {# [inline] extern \"rust-call\" fn call (& self , (c ,) : (char ,)) -> char :: EscapeDefault {{c . escape_default ()}}} impl  FnMut < (char ,) > for CharEscapeDefault {# [inline] extern \"rust-call\" fn call_mut (& mut self , (c ,) : (char ,)) -> char :: EscapeDefault {Fn :: call (&* self , (c ,))}} impl  FnOnce < (char ,) > for CharEscapeDefault {type Output = char :: EscapeDefault ; # [inline] extern \"rust-call\" fn call_once (self , (c ,) : (char ,)) -> char :: EscapeDefault {Fn :: call (& self , (c ,))}}");
    }

    #[test]
    fn test_impl_nonzero_fmt() {
        // from https://github.com/rust-lang/rust/blob/316a391dcb7d66dc25f1f9a4ec9d368ef7615005/src/libcore/num/mod.rs#L12
        let rules = create_rules(
            r#"
        macro_rules! impl_nonzero_fmt {
            ( #[$stability: meta] ( $( $Trait: ident ),+ ) for $Ty: ident ) => {
                fn foo() {}
            }
        }
"#,
        );

        assert_expansion(&rules, r#"impl_nonzero_fmt ! { # [ stable ( feature = "nonzero" , since = "1.28.0" ) ] ( Debug , Display , Binary , Octal , LowerHex , UpperHex ) for NonZeroU8 }"#, 
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

        assert_expansion(&rules, r#"__cfg_if_items ! { ( rustdoc , ) ; ( ( ) ( # [ cfg ( any ( target_os = "redox" , unix ) ) ] # [ stable ( feature = "rust1" , since = "1.0.0" ) ] pub use sys :: ext as unix ; # [ cfg ( windows ) ] # [ stable ( feature = "rust1" , since = "1.0.0" ) ] pub use sys :: ext as windows ; # [ cfg ( any ( target_os = "linux" , target_os = "l4re" ) ) ] pub mod linux ; ) ) , }"#,         
        "__cfg_if_items ! {(rustdoc , ) ; }");
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
            }
        }
"#,
        );

        assert_expansion(&rules, r#"
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
    }
}
