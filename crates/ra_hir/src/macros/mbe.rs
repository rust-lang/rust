use ra_syntax::SmolStr;

use crate::macros::tt;

#[derive(Debug)]
pub(crate) struct MacroRules {
    rules: Vec<Rule>,
}

#[derive(Debug)]
struct Rule {
    lhs: TokenTree,
    rhs: TokenTree,
}

#[derive(Debug)]
enum TokenTree {
    Leaf(Leaf),
    Subtree(Subtree),
    Repeat(Repeat),
}

#[derive(Debug)]
enum Leaf {
    Literal(Literal),
    Punct(Punct),
    Ident(Ident),
    Var(Var),
}

#[derive(Debug)]
struct Subtree {
    delimiter: Delimiter,
    token_trees: Vec<TokenTree>,
}

#[derive(Debug)]
enum Delimiter {
    Parenthesis,
    Brace,
    Bracket,
    None,
}

#[derive(Debug)]
struct Repeat {
    subtree: Subtree,
    kind: RepeatKind,
}

#[derive(Debug)]
enum RepeatKind {
    ZeroOrMore,
    OneOrMore,
    ZeroOrOne,
}

#[derive(Debug)]
struct Literal {
    text: SmolStr,
}

#[derive(Debug)]
struct Punct {
    char: char,
}

#[derive(Debug)]
struct Ident {
    text: SmolStr,
}

#[derive(Debug)]
struct Var {
    text: SmolStr,
}

pub(crate) fn parse(tt: &tt::Subtree) -> MacroRules {
    MacroRules { rules: Vec::new() }
}
