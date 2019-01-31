use smol_str::SmolStr;

#[derive(Debug)]
pub enum TokenTree {
    Leaf(Leaf),
    Subtree(Subtree),
}
impl_froms!(TokenTree: Leaf, Subtree);

#[derive(Debug)]
pub enum Leaf {
    Literal(Literal),
    Punct(Punct),
    Ident(Ident),
}
impl_froms!(Leaf: Literal, Punct, Ident);

#[derive(Debug)]
pub struct Subtree {
    pub delimiter: Delimiter,
    pub token_trees: Vec<TokenTree>,
}

#[derive(Clone, Copy, Debug)]
pub enum Delimiter {
    Parenthesis,
    Brace,
    Bracket,
    None,
}

#[derive(Debug)]
pub struct Literal {
    pub text: SmolStr,
}

#[derive(Debug)]
pub struct Punct {
    pub char: char,
}

#[derive(Debug)]
pub struct Ident {
    pub text: SmolStr,
}
