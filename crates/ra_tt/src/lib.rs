//! `tt` crate defines a `TokenTree` data structure: this is the interface (both
//! input and output) of macros. It closely mirrors `proc_macro` crate's
//! `TokenTree`.

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

use std::fmt;

use smol_str::SmolStr;

/// Represents identity of the token.
///
/// For hygiene purposes, we need to track which expanded tokens originated from
/// which source tokens. We do it by assigning an distinct identity to each
/// source token and making sure that identities are preserved during macro
/// expansion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TokenId(pub u32);

impl TokenId {
    pub const fn unspecified() -> TokenId {
        TokenId(!0)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TokenTree {
    Leaf(Leaf),
    Subtree(Subtree),
}
impl_froms!(TokenTree: Leaf, Subtree);

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Leaf {
    Literal(Literal),
    Punct(Punct),
    Ident(Ident),
}
impl_froms!(Leaf: Literal, Punct, Ident);

#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct Subtree {
    pub delimiter: Option<Delimiter>,
    pub token_trees: Vec<TokenTree>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Delimiter {
    pub id: TokenId,
    pub kind: DelimiterKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DelimiterKind {
    Parenthesis,
    Brace,
    Bracket,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Literal {
    pub text: SmolStr,
    pub id: TokenId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Punct {
    pub char: char,
    pub spacing: Spacing,
    pub id: TokenId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Spacing {
    Alone,
    Joint,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Ident {
    pub text: SmolStr,
    pub id: TokenId,
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
        let (l, r) = match self.delimiter_kind() {
            Some(DelimiterKind::Parenthesis) => ("(", ")"),
            Some(DelimiterKind::Brace) => ("{", "}"),
            Some(DelimiterKind::Bracket) => ("[", "]"),
            None => ("", ""),
        };
        f.write_str(l)?;
        let mut needs_space = false;
        for tt in self.token_trees.iter() {
            if needs_space {
                f.write_str(" ")?;
            }
            needs_space = true;
            match tt {
                TokenTree::Leaf(Leaf::Punct(p)) => {
                    needs_space = p.spacing == Spacing::Alone;
                    fmt::Display::fmt(p, f)?
                }
                tt => fmt::Display::fmt(tt, f)?,
            }
        }
        f.write_str(r)?;
        Ok(())
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

impl Subtree {
    /// Count the number of tokens recursively
    pub fn count(&self) -> usize {
        let children_count = self
            .token_trees
            .iter()
            .map(|c| match c {
                TokenTree::Subtree(c) => c.count(),
                _ => 0,
            })
            .sum::<usize>();

        self.token_trees.len() + children_count
    }

    pub fn delimiter_kind(&self) -> Option<DelimiterKind> {
        self.delimiter.map(|it| it.kind)
    }
}

pub mod buffer;
