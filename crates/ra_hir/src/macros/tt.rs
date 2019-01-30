use ra_syntax::SmolStr;

pub(crate) enum TokenTree {
    Leaf(Leaf),
    Subtree(Subtree),
}
impl_froms!(TokenTree: Leaf, Subtree);

pub(crate) enum Leaf {
    Literal(Literal),
    Punct(Punct),
    Ident(Ident),
}
impl_froms!(Leaf: Literal, Punct, Ident);

pub(crate) struct Subtree {
    pub(crate) delimiter: Delimiter,
    pub(crate) token_trees: Vec<TokenTree>,
}

pub(crate) enum Delimiter {
    Parenthesis,
    Brace,
    Bracket,
    None,
}

pub(crate) struct Literal {
    pub(crate) text: SmolStr,
}

pub(crate) struct Punct {
    pub(crate) char: char,
}

pub(crate) struct Ident {
    pub(crate) text: SmolStr,
}
