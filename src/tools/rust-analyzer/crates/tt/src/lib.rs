//! `tt` crate defines a `TokenTree` data structure: this is the interface (both
//! input and output) of macros. It closely mirrors `proc_macro` crate's
//! `TokenTree`.

#![warn(rust_2018_idioms, unused_lifetimes, semicolon_in_expressions_from_macros)]

use std::fmt;

use stdx::impl_from;

pub use smol_str::SmolStr;

/// Represents identity of the token.
///
/// For hygiene purposes, we need to track which expanded tokens originated from
/// which source tokens. We do it by assigning an distinct identity to each
/// source token and making sure that identities are preserved during macro
/// expansion.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct TokenId(pub u32);

impl fmt::Debug for TokenId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl TokenId {
    pub const UNSPECIFIED: TokenId = TokenId(!0);
    pub const fn unspecified() -> TokenId {
        Self::UNSPECIFIED
    }
}

pub mod token_id {
    pub use crate::{DelimiterKind, Spacing, TokenId};
    pub type Span = crate::TokenId;
    pub type Subtree = crate::Subtree<Span>;
    pub type Punct = crate::Punct<Span>;
    pub type Delimiter = crate::Delimiter<Span>;
    pub type Leaf = crate::Leaf<Span>;
    pub type Ident = crate::Ident<Span>;
    pub type Literal = crate::Literal<Span>;
    pub type TokenTree = crate::TokenTree<Span>;
    pub mod buffer {
        pub type TokenBuffer<'a> = crate::buffer::TokenBuffer<'a, super::Span>;
        pub type Cursor<'a> = crate::buffer::Cursor<'a, super::Span>;
        pub type TokenTreeRef<'a> = crate::buffer::TokenTreeRef<'a, super::Span>;
    }

    impl Delimiter {
        pub const UNSPECIFIED: Self = Self {
            open: TokenId::UNSPECIFIED,
            close: TokenId::UNSPECIFIED,
            kind: DelimiterKind::Invisible,
        };
        pub const fn unspecified() -> Self {
            Self::UNSPECIFIED
        }
    }
    impl Subtree {
        pub const fn empty() -> Self {
            Subtree { delimiter: Delimiter::unspecified(), token_trees: vec![] }
        }
    }
    impl TokenTree {
        pub const fn empty() -> Self {
            Self::Subtree(Subtree { delimiter: Delimiter::unspecified(), token_trees: vec![] })
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SyntaxContext(pub u32);

// #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
// pub struct Span {
//     pub id: TokenId,
//     pub ctx: SyntaxContext,
// }
// pub type Span = (TokenId, SyntaxContext);

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TokenTree<Span> {
    Leaf(Leaf<Span>),
    Subtree(Subtree<Span>),
}
impl_from!(Leaf<Span>, Subtree<Span> for TokenTree);

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Leaf<Span> {
    Literal(Literal<Span>),
    Punct(Punct<Span>),
    Ident(Ident<Span>),
}

impl<Span> Leaf<Span> {
    pub fn span(&self) -> &Span {
        match self {
            Leaf::Literal(it) => &it.span,
            Leaf::Punct(it) => &it.span,
            Leaf::Ident(it) => &it.span,
        }
    }
}
impl_from!(Literal<Span>, Punct<Span>, Ident<Span> for Leaf);

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Subtree<Span> {
    // FIXME, this should not be Option
    pub delimiter: Delimiter<Span>,
    pub token_trees: Vec<TokenTree<Span>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Delimiter<Span> {
    pub open: Span,
    pub close: Span,
    pub kind: DelimiterKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DelimiterKind {
    Parenthesis,
    Brace,
    Bracket,
    Invisible,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Literal<Span> {
    pub text: SmolStr,
    pub span: Span,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Punct<Span> {
    pub char: char,
    pub spacing: Spacing,
    pub span: Span,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Spacing {
    Alone,
    Joint,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
/// Identifier or keyword. Unlike rustc, we keep "r#" prefix when it represents a raw identifier.
pub struct Ident<Span> {
    pub text: SmolStr,
    pub span: Span,
}

fn print_debug_subtree<Span: fmt::Debug>(
    f: &mut fmt::Formatter<'_>,
    subtree: &Subtree<Span>,
    level: usize,
) -> fmt::Result {
    let align = "  ".repeat(level);

    let Delimiter { kind, open, close } = &subtree.delimiter;
    let aux = match kind {
        DelimiterKind::Invisible => format!("$$ {:?} {:?}", open, close),
        DelimiterKind::Parenthesis => format!("() {:?} {:?}", open, close),
        DelimiterKind::Brace => format!("{{}} {:?} {:?}", open, close),
        DelimiterKind::Bracket => format!("[] {:?} {:?}", open, close),
    };

    if subtree.token_trees.is_empty() {
        write!(f, "{align}SUBTREE {aux}")?;
    } else {
        writeln!(f, "{align}SUBTREE {aux}")?;
        for (idx, child) in subtree.token_trees.iter().enumerate() {
            print_debug_token(f, child, level + 1)?;
            if idx != subtree.token_trees.len() - 1 {
                writeln!(f)?;
            }
        }
    }

    Ok(())
}

fn print_debug_token<Span: fmt::Debug>(
    f: &mut fmt::Formatter<'_>,
    tkn: &TokenTree<Span>,
    level: usize,
) -> fmt::Result {
    let align = "  ".repeat(level);

    match tkn {
        TokenTree::Leaf(leaf) => match leaf {
            Leaf::Literal(lit) => write!(f, "{}LITERAL {} {:?}", align, lit.text, lit.span)?,
            Leaf::Punct(punct) => write!(
                f,
                "{}PUNCH   {} [{}] {:?}",
                align,
                punct.char,
                if punct.spacing == Spacing::Alone { "alone" } else { "joint" },
                punct.span
            )?,
            Leaf::Ident(ident) => write!(f, "{}IDENT   {} {:?}", align, ident.text, ident.span)?,
        },
        TokenTree::Subtree(subtree) => {
            print_debug_subtree(f, subtree, level)?;
        }
    }

    Ok(())
}

impl<Span: fmt::Debug> fmt::Debug for Subtree<Span> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        print_debug_subtree(f, self, 0)
    }
}

impl<Span> fmt::Display for TokenTree<Span> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TokenTree::Leaf(it) => fmt::Display::fmt(it, f),
            TokenTree::Subtree(it) => fmt::Display::fmt(it, f),
        }
    }
}

impl<Span> fmt::Display for Subtree<Span> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (l, r) = match self.delimiter.kind {
            DelimiterKind::Parenthesis => ("(", ")"),
            DelimiterKind::Brace => ("{", "}"),
            DelimiterKind::Bracket => ("[", "]"),
            DelimiterKind::Invisible => ("", ""),
        };
        f.write_str(l)?;
        let mut needs_space = false;
        for tt in &self.token_trees {
            if needs_space {
                f.write_str(" ")?;
            }
            needs_space = true;
            match tt {
                TokenTree::Leaf(Leaf::Punct(p)) => {
                    needs_space = p.spacing == Spacing::Alone;
                    fmt::Display::fmt(p, f)?;
                }
                tt => fmt::Display::fmt(tt, f)?,
            }
        }
        f.write_str(r)?;
        Ok(())
    }
}

impl<Span> fmt::Display for Leaf<Span> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Leaf::Ident(it) => fmt::Display::fmt(it, f),
            Leaf::Literal(it) => fmt::Display::fmt(it, f),
            Leaf::Punct(it) => fmt::Display::fmt(it, f),
        }
    }
}

impl<Span> fmt::Display for Ident<Span> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.text, f)
    }
}

impl<Span> fmt::Display for Literal<Span> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.text, f)
    }
}

impl<Span> fmt::Display for Punct<Span> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.char, f)
    }
}

impl<Span> Subtree<Span> {
    /// Count the number of tokens recursively
    pub fn count(&self) -> usize {
        let children_count = self
            .token_trees
            .iter()
            .map(|c| match c {
                TokenTree::Subtree(c) => c.count(),
                TokenTree::Leaf(_) => 0,
            })
            .sum::<usize>();

        self.token_trees.len() + children_count
    }
}

impl<Span> Subtree<Span> {
    /// A simple line string used for debugging
    pub fn as_debug_string(&self) -> String {
        let delim = match self.delimiter.kind {
            DelimiterKind::Brace => ("{", "}"),
            DelimiterKind::Bracket => ("[", "]"),
            DelimiterKind::Parenthesis => ("(", ")"),
            DelimiterKind::Invisible => ("$", "$"),
        };

        let mut res = String::new();
        res.push_str(delim.0);
        let mut last = None;
        for child in &self.token_trees {
            let s = match child {
                TokenTree::Leaf(it) => {
                    let s = match it {
                        Leaf::Literal(it) => it.text.to_string(),
                        Leaf::Punct(it) => it.char.to_string(),
                        Leaf::Ident(it) => it.text.to_string(),
                    };
                    match (it, last) {
                        (Leaf::Ident(_), Some(&TokenTree::Leaf(Leaf::Ident(_)))) => {
                            " ".to_string() + &s
                        }
                        (Leaf::Punct(_), Some(TokenTree::Leaf(Leaf::Punct(punct)))) => {
                            if punct.spacing == Spacing::Alone {
                                " ".to_string() + &s
                            } else {
                                s
                            }
                        }
                        _ => s,
                    }
                }
                TokenTree::Subtree(it) => it.as_debug_string(),
            };
            res.push_str(&s);
            last = Some(child);
        }

        res.push_str(delim.1);
        res
    }
}

pub mod buffer;

pub fn pretty<Span>(tkns: &[TokenTree<Span>]) -> String {
    fn tokentree_to_text<Span>(tkn: &TokenTree<Span>) -> String {
        match tkn {
            TokenTree::Leaf(Leaf::Ident(ident)) => ident.text.clone().into(),
            TokenTree::Leaf(Leaf::Literal(literal)) => literal.text.clone().into(),
            TokenTree::Leaf(Leaf::Punct(punct)) => format!("{}", punct.char),
            TokenTree::Subtree(subtree) => {
                let content = pretty(&subtree.token_trees);
                let (open, close) = match subtree.delimiter.kind {
                    DelimiterKind::Brace => ("{", "}"),
                    DelimiterKind::Bracket => ("[", "]"),
                    DelimiterKind::Parenthesis => ("(", ")"),
                    DelimiterKind::Invisible => ("", ""),
                };
                format!("{open}{content}{close}")
            }
        }
    }

    tkns.iter()
        .fold((String::new(), true), |(last, last_to_joint), tkn| {
            let s = [last, tokentree_to_text(tkn)].join(if last_to_joint { "" } else { " " });
            let mut is_joint = false;
            if let TokenTree::Leaf(Leaf::Punct(punct)) = tkn {
                if punct.spacing == Spacing::Joint {
                    is_joint = true;
                }
            }
            (s, is_joint)
        })
        .0
}
