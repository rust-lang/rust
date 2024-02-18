//! `tt` crate defines a `TokenTree` data structure: this is the interface (both
//! input and output) of macros. It closely mirrors `proc_macro` crate's
//! `TokenTree`.

#![warn(rust_2018_idioms, unused_lifetimes)]

use std::fmt;

use stdx::impl_from;

pub use smol_str::SmolStr;
pub use text_size::{TextRange, TextSize};

pub trait Span: std::fmt::Debug + Copy + Sized + Eq {}

impl<Ctx> Span for span::SpanData<Ctx> where span::SpanData<Ctx>: std::fmt::Debug + Copy + Sized + Eq
{}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TokenTree<S> {
    Leaf(Leaf<S>),
    Subtree(Subtree<S>),
}
impl_from!(Leaf<S>, Subtree<S> for TokenTree);
impl<S: Span> TokenTree<S> {
    pub fn empty(span: S) -> Self {
        Self::Subtree(Subtree {
            delimiter: Delimiter::invisible_spanned(span),
            token_trees: Box::new([]),
        })
    }

    pub fn subtree_or_wrap(self, span: DelimSpan<S>) -> Subtree<S> {
        match self {
            TokenTree::Leaf(_) => Subtree {
                delimiter: Delimiter::invisible_delim_spanned(span),
                token_trees: Box::new([self]),
            },
            TokenTree::Subtree(s) => s,
        }
    }

    pub fn first_span(&self) -> S {
        match self {
            TokenTree::Leaf(l) => *l.span(),
            TokenTree::Subtree(s) => s.delimiter.open,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Leaf<S> {
    Literal(Literal<S>),
    Punct(Punct<S>),
    Ident(Ident<S>),
}

impl<S> Leaf<S> {
    pub fn span(&self) -> &S {
        match self {
            Leaf::Literal(it) => &it.span,
            Leaf::Punct(it) => &it.span,
            Leaf::Ident(it) => &it.span,
        }
    }
}
impl_from!(Literal<S>, Punct<S>, Ident<S> for Leaf);

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Subtree<S> {
    pub delimiter: Delimiter<S>,
    pub token_trees: Box<[TokenTree<S>]>,
}

impl<S: Span> Subtree<S> {
    pub fn empty(span: DelimSpan<S>) -> Self {
        Subtree { delimiter: Delimiter::invisible_delim_spanned(span), token_trees: Box::new([]) }
    }

    /// This is slow, and should be avoided, as it will always reallocate!
    pub fn push(&mut self, subtree: TokenTree<S>) {
        let mut mutable_trees = std::mem::take(&mut self.token_trees).into_vec();

        // Reserve exactly space for one element, to avoid `into_boxed_slice` having to reallocate again.
        mutable_trees.reserve_exact(1);
        mutable_trees.push(subtree);

        self.token_trees = mutable_trees.into_boxed_slice();
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct SubtreeBuilder<S> {
    pub delimiter: Delimiter<S>,
    pub token_trees: Vec<TokenTree<S>>,
}

impl<S> SubtreeBuilder<S> {
    pub fn build(self) -> Subtree<S> {
        Subtree { delimiter: self.delimiter, token_trees: self.token_trees.into_boxed_slice() }
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct DelimSpan<S> {
    pub open: S,
    pub close: S,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Delimiter<S> {
    pub open: S,
    pub close: S,
    pub kind: DelimiterKind,
}

impl<S: Span> Delimiter<S> {
    pub const fn invisible_spanned(span: S) -> Self {
        Delimiter { open: span, close: span, kind: DelimiterKind::Invisible }
    }

    pub const fn invisible_delim_spanned(span: DelimSpan<S>) -> Self {
        Delimiter { open: span.open, close: span.close, kind: DelimiterKind::Invisible }
    }

    pub fn delim_span(&self) -> DelimSpan<S> {
        DelimSpan { open: self.open, close: self.close }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DelimiterKind {
    Parenthesis,
    Brace,
    Bracket,
    Invisible,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Literal<S> {
    pub text: SmolStr,
    pub span: S,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Punct<S> {
    pub char: char,
    pub spacing: Spacing,
    pub span: S,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Spacing {
    Alone,
    /// Whether the following token is joint to the current one.
    Joint,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
/// Identifier or keyword. Unlike rustc, we keep "r#" prefix when it represents a raw identifier.
pub struct Ident<S> {
    pub text: SmolStr,
    pub span: S,
}

impl<S> Ident<S> {
    pub fn new(text: impl Into<SmolStr>, span: S) -> Self {
        Ident { text: text.into(), span }
    }
}

fn print_debug_subtree<S: fmt::Debug>(
    f: &mut fmt::Formatter<'_>,
    subtree: &Subtree<S>,
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

fn print_debug_token<S: fmt::Debug>(
    f: &mut fmt::Formatter<'_>,
    tkn: &TokenTree<S>,
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

impl<S: fmt::Debug> fmt::Debug for Subtree<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        print_debug_subtree(f, self, 0)
    }
}

impl<S> fmt::Display for TokenTree<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TokenTree::Leaf(it) => fmt::Display::fmt(it, f),
            TokenTree::Subtree(it) => fmt::Display::fmt(it, f),
        }
    }
}

impl<S> fmt::Display for Subtree<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (l, r) = match self.delimiter.kind {
            DelimiterKind::Parenthesis => ("(", ")"),
            DelimiterKind::Brace => ("{", "}"),
            DelimiterKind::Bracket => ("[", "]"),
            DelimiterKind::Invisible => ("", ""),
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
                    fmt::Display::fmt(p, f)?;
                }
                tt => fmt::Display::fmt(tt, f)?,
            }
        }
        f.write_str(r)?;
        Ok(())
    }
}

impl<S> fmt::Display for Leaf<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Leaf::Ident(it) => fmt::Display::fmt(it, f),
            Leaf::Literal(it) => fmt::Display::fmt(it, f),
            Leaf::Punct(it) => fmt::Display::fmt(it, f),
        }
    }
}

impl<S> fmt::Display for Ident<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.text, f)
    }
}

impl<S> fmt::Display for Literal<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.text, f)
    }
}

impl<S> fmt::Display for Punct<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.char, f)
    }
}

impl<S> Subtree<S> {
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

impl<S> Subtree<S> {
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
        for child in self.token_trees.iter() {
            let s = match child {
                TokenTree::Leaf(it) => {
                    let s = match it {
                        Leaf::Literal(it) => it.text.to_string(),
                        Leaf::Punct(it) => it.char.to_string(),
                        Leaf::Ident(it) => it.text.to_string(),
                    };
                    match (it, last) {
                        (Leaf::Ident(_), Some(&TokenTree::Leaf(Leaf::Ident(_)))) => {
                            " ".to_owned() + &s
                        }
                        (Leaf::Punct(_), Some(TokenTree::Leaf(Leaf::Punct(punct)))) => {
                            if punct.spacing == Spacing::Alone {
                                " ".to_owned() + &s
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

pub fn pretty<S>(tkns: &[TokenTree<S>]) -> String {
    fn tokentree_to_text<S>(tkn: &TokenTree<S>) -> String {
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
