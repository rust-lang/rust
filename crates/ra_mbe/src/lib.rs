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

pub use crate::syntax_bridge::ast_to_token_tree;

/// This struct contains AST for a single `macro_rules` defenition. What might
/// be very confusing is that AST has almost exactly the same shape as
/// `tt::TokenTree`, but there's a crucial difference: in macro rules, `$ident`
/// and `$()*` have special meaning (see `Var` and `Repeat` data structures)
#[derive(Debug)]
pub struct MacroRules {
    pub(crate) rules: Vec<Rule>,
}

impl MacroRules {
    pub fn parse(tt: &tt::Subtree) -> Option<MacroRules> {
        mbe_parser::parse(tt)
    }
    pub fn expand(&self, tt: &tt::Subtree) -> Option<tt::Subtree> {
        mbe_expander::exapnd(self, tt)
    }
}

#[derive(Debug)]
pub(crate) struct Rule {
    pub(crate) lhs: Subtree,
    pub(crate) rhs: Subtree,
}

#[derive(Debug)]
pub(crate) enum TokenTree {
    Leaf(Leaf),
    Subtree(Subtree),
    Repeat(Repeat),
}
impl_froms!(TokenTree: Leaf, Subtree, Repeat);

#[derive(Debug)]
pub(crate) enum Leaf {
    Literal(Literal),
    Punct(Punct),
    Ident(Ident),
    Var(Var),
}
impl_froms!(Leaf: Literal, Punct, Ident, Var);

#[derive(Debug)]
pub(crate) struct Subtree {
    pub(crate) delimiter: Delimiter,
    pub(crate) token_trees: Vec<TokenTree>,
}

#[derive(Debug)]
pub(crate) struct Repeat {
    pub(crate) subtree: Subtree,
    pub(crate) kind: RepeatKind,
    pub(crate) separator: Option<char>,
}

#[derive(Debug)]
pub(crate) enum RepeatKind {
    ZeroOrMore,
    OneOrMore,
    ZeroOrOne,
}

#[derive(Debug)]
pub(crate) struct Literal {
    pub(crate) text: SmolStr,
}

#[derive(Debug)]
pub(crate) struct Ident {
    pub(crate) text: SmolStr,
}

#[derive(Debug)]
pub(crate) struct Var {
    pub(crate) text: SmolStr,
    pub(crate) kind: Option<SmolStr>,
}

#[cfg(test)]
mod tests {
    use ra_syntax::{ast, AstNode};

    use super::*;

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
        let macro_definition = source_file
            .syntax()
            .descendants()
            .find_map(ast::MacroCall::cast)
            .unwrap();

        let source_file = ast::SourceFile::parse(macro_invocation);
        let macro_invocation = source_file
            .syntax()
            .descendants()
            .find_map(ast::MacroCall::cast)
            .unwrap();

        let definition_tt = ast_to_token_tree(macro_definition.token_tree().unwrap()).unwrap();
        let invocation_tt = ast_to_token_tree(macro_invocation.token_tree().unwrap()).unwrap();
        let rules = crate::MacroRules::parse(&definition_tt).unwrap();
        let expansion = rules.expand(&invocation_tt).unwrap();
        assert_eq!(
        expansion.to_string(),
        "impl From < Leaf > for TokenTree {fn from (it : Leaf) -> TokenTree {TokenTree :: Leaf (it)}} \
         impl From < Subtree > for TokenTree {fn from (it : Subtree) -> TokenTree {TokenTree :: Subtree (it)}}"
    )
    }
}
