use ra_syntax::SmolStr;

pub(crate) enum TokenTree {
    Leaf(Leaf),
    Subtree(Subtree),
}

pub(crate) enum Leaf {
    Literal(Literal),
    Punct(Punct),
    Ident(Ident),
}

pub(crate) struct Subtree {
    delimiter: Delimiter,
    token_trees: Vec<TokenTree>,
}

pub(crate) enum Delimiter {
    Parenthesis,
    Brace,
    Bracket,
    None,
}

pub(crate) struct Literal {
    text: SmolStr,
}

pub(crate) struct Punct {
    char: char,
}

pub(crate) struct Ident {
    text: SmolStr,
}
