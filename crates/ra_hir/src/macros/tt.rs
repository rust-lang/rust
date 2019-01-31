use ra_syntax::SmolStr;

#[derive(Debug)]
pub(crate) enum TokenTree {
    Leaf(Leaf),
    Subtree(Subtree),
}
impl_froms!(TokenTree: Leaf, Subtree);

#[derive(Debug)]
pub(crate) enum Leaf {
    Literal(Literal),
    Punct(Punct),
    Ident(Ident),
}
impl_froms!(Leaf: Literal, Punct, Ident);

#[derive(Debug)]
pub(crate) struct Subtree {
    pub(crate) delimiter: Delimiter,
    pub(crate) token_trees: Vec<TokenTree>,
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum Delimiter {
    Parenthesis,
    Brace,
    Bracket,
    None,
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
