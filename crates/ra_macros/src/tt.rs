use std::fmt;

use smol_str::SmolStr;
use join_to_string::join;

#[derive(Debug, Clone)]
pub enum TokenTree {
    Leaf(Leaf),
    Subtree(Subtree),
}
impl_froms!(TokenTree: Leaf, Subtree);

#[derive(Debug, Clone)]
pub enum Leaf {
    Literal(Literal),
    Punct(Punct),
    Ident(Ident),
}
impl_froms!(Leaf: Literal, Punct, Ident);

#[derive(Debug, Clone)]
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

#[derive(Debug, Clone)]
pub struct Literal {
    pub text: SmolStr,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Punct {
    pub char: char,
}

#[derive(Debug, Clone)]
pub struct Ident {
    pub text: SmolStr,
}

impl fmt::Display for TokenTree {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            TokenTree::Leaf(it) => fmt::Display::fmt(it, f),
            TokenTree::Subtree(it) => fmt::Display::fmt(it, f),
        }
    }
}

impl fmt::Display for Subtree {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let (l, r) = match self.delimiter {
            Delimiter::Parenthesis => ("(", ")"),
            Delimiter::Brace => ("{", "}"),
            Delimiter::Bracket => ("[", "]"),
            Delimiter::None => ("", ""),
        };
        join(self.token_trees.iter())
            .separator(" ")
            .surround_with(l, r)
            .to_fmt(f)
    }
}

impl fmt::Display for Leaf {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Leaf::Ident(it) => fmt::Display::fmt(it, f),
            Leaf::Literal(it) => fmt::Display::fmt(it, f),
            Leaf::Punct(it) => fmt::Display::fmt(it, f),
        }
    }
}

impl fmt::Display for Ident {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(&self.text, f)
    }
}

impl fmt::Display for Literal {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(&self.text, f)
    }
}

impl fmt::Display for Punct {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(&self.char, f)
    }
}
