//! This module implements declarative macros: old `macro_rules` and the newer
//! `macro`. Declarative macros are also known as "macro by example", and that's
//! why we call this module `mbe`. For external documentation, prefer the
//! official terminology: "declarative macros".

crate mod macro_check;
crate mod macro_parser;
crate mod macro_rules;
crate mod quoted;
crate mod transcribe;

use rustc_ast::token::{self, NonterminalKind, Token, TokenKind};
use rustc_ast::tokenstream::DelimSpan;

use rustc_span::symbol::Ident;
use rustc_span::Span;

use rustc_data_structures::sync::Lrc;

/// Contains the sub-token-trees of a "delimited" token tree, such as the contents of `(`. Note
/// that the delimiter itself might be `NoDelim`.
#[derive(Clone, PartialEq, Encodable, Decodable, Debug)]
struct Delimited {
    delim: token::DelimToken,
    tts: Vec<TokenTree>,
}

impl Delimited {
    /// Returns a `self::TokenTree` with a `Span` corresponding to the opening delimiter.
    fn open_tt(&self, span: DelimSpan) -> TokenTree {
        TokenTree::token(token::OpenDelim(self.delim), span.open)
    }

    /// Returns a `self::TokenTree` with a `Span` corresponding to the closing delimiter.
    fn close_tt(&self, span: DelimSpan) -> TokenTree {
        TokenTree::token(token::CloseDelim(self.delim), span.close)
    }
}

#[derive(Clone, PartialEq, Encodable, Decodable, Debug)]
struct SequenceRepetition {
    /// The sequence of token trees
    tts: Vec<TokenTree>,
    /// The optional separator
    separator: Option<Token>,
    /// Whether the sequence can be repeated zero (*), or one or more times (+)
    kleene: KleeneToken,
    /// The number of `Match`s that appear in the sequence (and subsequences)
    num_captures: usize,
}

#[derive(Clone, PartialEq, Encodable, Decodable, Debug, Copy)]
struct KleeneToken {
    span: Span,
    op: KleeneOp,
}

impl KleeneToken {
    fn new(op: KleeneOp, span: Span) -> KleeneToken {
        KleeneToken { span, op }
    }
}

/// A Kleene-style [repetition operator](https://en.wikipedia.org/wiki/Kleene_star)
/// for token sequences.
#[derive(Clone, PartialEq, Encodable, Decodable, Debug, Copy)]
enum KleeneOp {
    /// Kleene star (`*`) for zero or more repetitions
    ZeroOrMore,
    /// Kleene plus (`+`) for one or more repetitions
    OneOrMore,
    /// Kleene optional (`?`) for zero or one reptitions
    ZeroOrOne,
}

/// Similar to `tokenstream::TokenTree`, except that `$i`, `$i:ident`, and `$(...)`
/// are "first-class" token trees. Useful for parsing macros.
#[derive(Debug, Clone, PartialEq, Encodable, Decodable)]
enum TokenTree {
    Token(Token),
    Delimited(DelimSpan, Lrc<Delimited>),
    /// A kleene-style repetition sequence
    Sequence(DelimSpan, Lrc<SequenceRepetition>),
    /// e.g., `$var`
    MetaVar(Span, Ident),
    /// e.g., `$var:expr`. This is only used in the left hand side of MBE macros.
    MetaVarDecl(Span, Ident /* name to bind */, Option<NonterminalKind>),
}

impl TokenTree {
    /// Return the number of tokens in the tree.
    fn len(&self) -> usize {
        match *self {
            TokenTree::Delimited(_, ref delimed) => match delimed.delim {
                token::NoDelim => delimed.tts.len(),
                _ => delimed.tts.len() + 2,
            },
            TokenTree::Sequence(_, ref seq) => seq.tts.len(),
            _ => 0,
        }
    }

    /// Returns `true` if the given token tree is delimited.
    fn is_delimited(&self) -> bool {
        matches!(*self, TokenTree::Delimited(..))
    }

    /// Returns `true` if the given token tree is a token of the given kind.
    fn is_token(&self, expected_kind: &TokenKind) -> bool {
        match self {
            TokenTree::Token(Token { kind: actual_kind, .. }) => actual_kind == expected_kind,
            _ => false,
        }
    }

    /// Gets the `index`-th sub-token-tree. This only makes sense for delimited trees and sequences.
    fn get_tt(&self, index: usize) -> TokenTree {
        match (self, index) {
            (&TokenTree::Delimited(_, ref delimed), _) if delimed.delim == token::NoDelim => {
                delimed.tts[index].clone()
            }
            (&TokenTree::Delimited(span, ref delimed), _) => {
                if index == 0 {
                    return delimed.open_tt(span);
                }
                if index == delimed.tts.len() + 1 {
                    return delimed.close_tt(span);
                }
                delimed.tts[index - 1].clone()
            }
            (&TokenTree::Sequence(_, ref seq), _) => seq.tts[index].clone(),
            _ => panic!("Cannot expand a token tree"),
        }
    }

    /// Retrieves the `TokenTree`'s span.
    fn span(&self) -> Span {
        match *self {
            TokenTree::Token(Token { span, .. })
            | TokenTree::MetaVar(span, _)
            | TokenTree::MetaVarDecl(span, _, _) => span,
            TokenTree::Delimited(span, _) | TokenTree::Sequence(span, _) => span.entire(),
        }
    }

    fn token(kind: TokenKind, span: Span) -> TokenTree {
        TokenTree::Token(Token::new(kind, span))
    }
}
