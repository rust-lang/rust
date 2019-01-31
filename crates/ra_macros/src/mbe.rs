use smol_str::SmolStr;

use crate::tt::Delimiter;

pub use crate::{
    mbe_parser::parse,
    mbe_expander::exapnd,
};

#[derive(Debug)]
pub struct MacroRules {
    pub(crate) rules: Vec<Rule>,
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
    pub(crate) separator: Option<Punct>,
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
pub(crate) struct Punct {
    pub(crate) char: char,
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
