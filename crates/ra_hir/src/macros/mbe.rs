use ra_syntax::SmolStr;

use crate::macros::tt;

struct MacroRules {
    rules: Vec<Rule>,
}

struct Rule {
    lhs: TokenTree,
    rhs: TokenTree,
}

enum TokenTree {
    Leaf(Leaf),
    Subtree(Subtree),
    Repeat(Repeat),
}

enum Leaf {
    Literal(Literal),
    Punct(Punct),
    Ident(Ident),
    Var(Var),
}

struct Subtree {
    delimiter: Delimiter,
    token_trees: Vec<TokenTree>,
}

enum Delimiter {
    Parenthesis,
    Brace,
    Bracket,
    None,
}

struct Repeat {
    subtree: Subtree,
    kind: RepeatKind,
}

enum RepeatKind {
    ZeroOrMore,
    OneOrMore,
    ZeroOrOne,
}

struct Literal {
    text: SmolStr,
}

struct Punct {
    char: char,
}

struct Ident {
    text: SmolStr,
}

struct Var {
    text: SmolStr,
}

fn parse(tt: tt::TokenTree) -> MacroRules {
    MacroRules { rules: Vec::new() }
}
