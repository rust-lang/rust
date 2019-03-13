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

mod tt_cursor;
mod mbe_parser;
mod mbe_expander;
mod syntax_bridge;

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
}

pub use crate::syntax_bridge::{ast_to_token_tree, token_tree_to_ast_item_list};

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

    fn create_rules(macro_definition: &str) -> MacroRules {
        let source_file = ast::SourceFile::parse(macro_definition);
        let macro_definition =
            source_file.syntax().descendants().find_map(ast::MacroCall::cast).unwrap();

        let (definition_tt, _) = ast_to_token_tree(macro_definition.token_tree().unwrap()).unwrap();
        crate::MacroRules::parse(&definition_tt).unwrap()
    }

    fn expand(rules: &MacroRules, invocation: &str) -> tt::Subtree {
        let source_file = ast::SourceFile::parse(invocation);
        let macro_invocation =
            source_file.syntax().descendants().find_map(ast::MacroCall::cast).unwrap();

        let (invocation_tt, _) = ast_to_token_tree(macro_invocation.token_tree().unwrap()).unwrap();

        rules.expand(&invocation_tt).unwrap()
    }

    fn assert_expansion(rules: &MacroRules, invocation: &str, expansion: &str) {
        let expanded = expand(rules, invocation);
        assert_eq!(expanded.to_string(), expansion);
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
        let tree = token_tree_to_ast_item_list(&expansion);
        assert_eq!(
            tree.syntax().debug_dump().trim(),
            r#"
SOURCE_FILE@[0; 40)
  STRUCT_DEF@[0; 20)
    STRUCT_KW@[0; 6)
    NAME@[6; 9)
      IDENT@[6; 9) "Foo"
    NAMED_FIELD_DEF_LIST@[9; 20)
      L_CURLY@[9; 10)
      NAMED_FIELD_DEF@[10; 19)
        NAME@[10; 15)
          IDENT@[10; 15) "field"
        COLON@[15; 16)
        PATH_TYPE@[16; 19)
          PATH@[16; 19)
            PATH_SEGMENT@[16; 19)
              NAME_REF@[16; 19)
                IDENT@[16; 19) "u32"
      R_CURLY@[19; 20)
  STRUCT_DEF@[20; 40)
    STRUCT_KW@[20; 26)
    NAME@[26; 29)
      IDENT@[26; 29) "Bar"
    NAMED_FIELD_DEF_LIST@[29; 40)
      L_CURLY@[29; 30)
      NAMED_FIELD_DEF@[30; 39)
        NAME@[30; 35)
          IDENT@[30; 35) "field"
        COLON@[35; 36)
        PATH_TYPE@[36; 39)
          PATH@[36; 39)
            PATH_SEGMENT@[36; 39)
              NAME_REF@[36; 39)
                IDENT@[36; 39) "u32"
      R_CURLY@[39; 40)"#
                .trim()
        );
    }

}
