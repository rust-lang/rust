//! Our parser is generic over the source of tokens it parses.
//!
//! This module defines tokens sourced from declarative macros.

use parser::{Token, TokenSource};
use syntax::{lex_single_syntax_kind, SmolStr, SyntaxKind, SyntaxKind::*, T};
use tt::buffer::TokenBuffer;

#[derive(Debug, Clone, Eq, PartialEq)]
struct TtToken {
    tt: Token,
    text: SmolStr,
}

pub(crate) struct SubtreeTokenSource {
    cached: Vec<TtToken>,
    curr: (Token, usize),
}

impl<'a> SubtreeTokenSource {
    // Helper function used in test
    #[cfg(test)]
    pub(crate) fn text(&self) -> SmolStr {
        match self.cached.get(self.curr.1) {
            Some(tt) => tt.text.clone(),
            _ => SmolStr::new(""),
        }
    }
}

impl<'a> SubtreeTokenSource {
    pub(crate) fn new(buffer: &TokenBuffer) -> SubtreeTokenSource {
        let mut current = buffer.begin();
        let mut cached = Vec::with_capacity(100);

        while !current.eof() {
            let cursor = current;
            let tt = cursor.token_tree();

            // Check if it is lifetime
            if let Some(tt::buffer::TokenTreeRef::Leaf(tt::Leaf::Punct(punct), _)) = tt {
                if punct.char == '\'' {
                    let next = cursor.bump();
                    if let Some(tt::buffer::TokenTreeRef::Leaf(tt::Leaf::Ident(ident), _)) =
                        next.token_tree()
                    {
                        let text = SmolStr::new("'".to_string() + &ident.text);
                        cached.push(TtToken {
                            tt: Token { kind: LIFETIME_IDENT, is_jointed_to_next: false },
                            text,
                        });
                        current = next.bump();
                        continue;
                    } else {
                        panic!("Next token must be ident : {:#?}", next.token_tree());
                    }
                }
            }

            current = match tt {
                Some(tt::buffer::TokenTreeRef::Leaf(leaf, _)) => {
                    cached.push(convert_leaf(leaf));
                    cursor.bump()
                }
                Some(tt::buffer::TokenTreeRef::Subtree(subtree, _)) => {
                    cached.push(convert_delim(subtree.delimiter_kind(), false));
                    cursor.subtree().unwrap()
                }
                None => {
                    if let Some(subtree) = cursor.end() {
                        cached.push(convert_delim(subtree.delimiter_kind(), true));
                        cursor.bump()
                    } else {
                        continue;
                    }
                }
            };
        }

        let mut res = SubtreeTokenSource {
            curr: (Token { kind: EOF, is_jointed_to_next: false }, 0),
            cached,
        };
        res.curr = (res.token(0), 0);
        res
    }

    fn token(&self, pos: usize) -> Token {
        match self.cached.get(pos) {
            Some(it) => it.tt,
            None => Token { kind: EOF, is_jointed_to_next: false },
        }
    }
}

impl<'a> TokenSource for SubtreeTokenSource {
    fn current(&self) -> Token {
        self.curr.0
    }

    /// Lookahead n token
    fn lookahead_nth(&self, n: usize) -> Token {
        self.token(self.curr.1 + n)
    }

    /// bump cursor to next token
    fn bump(&mut self) {
        if self.current().kind == EOF {
            return;
        }
        self.curr = (self.token(self.curr.1 + 1), self.curr.1 + 1);
    }

    /// Is the current token a specified keyword?
    fn is_keyword(&self, kw: &str) -> bool {
        match self.cached.get(self.curr.1) {
            Some(t) => t.text == *kw,
            None => false,
        }
    }
}

fn convert_delim(d: Option<tt::DelimiterKind>, closing: bool) -> TtToken {
    let (kinds, texts) = match d {
        Some(tt::DelimiterKind::Parenthesis) => ([T!['('], T![')']], "()"),
        Some(tt::DelimiterKind::Brace) => ([T!['{'], T!['}']], "{}"),
        Some(tt::DelimiterKind::Bracket) => ([T!['['], T![']']], "[]"),
        None => ([L_DOLLAR, R_DOLLAR], ""),
    };

    let idx = closing as usize;
    let kind = kinds[idx];
    let text = if !texts.is_empty() { &texts[idx..texts.len() - (1 - idx)] } else { "" };
    TtToken { tt: Token { kind, is_jointed_to_next: false }, text: SmolStr::new(text) }
}

fn convert_literal(l: &tt::Literal) -> TtToken {
    let is_negated = l.text.starts_with('-');
    let inner_text = &l.text[if is_negated { 1 } else { 0 }..];

    let kind = lex_single_syntax_kind(inner_text)
        .map(|(kind, _error)| kind)
        .filter(|kind| {
            kind.is_literal() && (!is_negated || matches!(kind, FLOAT_NUMBER | INT_NUMBER))
        })
        .unwrap_or_else(|| panic!("Fail to convert given literal {:#?}", &l));

    TtToken { tt: Token { kind, is_jointed_to_next: false }, text: l.text.clone() }
}

fn convert_ident(ident: &tt::Ident) -> TtToken {
    let kind = match ident.text.as_ref() {
        "true" => T![true],
        "false" => T![false],
        "_" => UNDERSCORE,
        i if i.starts_with('\'') => LIFETIME_IDENT,
        _ => SyntaxKind::from_keyword(ident.text.as_str()).unwrap_or(IDENT),
    };

    TtToken { tt: Token { kind, is_jointed_to_next: false }, text: ident.text.clone() }
}

fn convert_punct(p: tt::Punct) -> TtToken {
    let kind = match SyntaxKind::from_char(p.char) {
        None => panic!("{:#?} is not a valid punct", p),
        Some(kind) => kind,
    };

    let text = {
        let mut buf = [0u8; 4];
        let s: &str = p.char.encode_utf8(&mut buf);
        SmolStr::new(s)
    };
    TtToken { tt: Token { kind, is_jointed_to_next: p.spacing == tt::Spacing::Joint }, text }
}

fn convert_leaf(leaf: &tt::Leaf) -> TtToken {
    match leaf {
        tt::Leaf::Literal(l) => convert_literal(l),
        tt::Leaf::Ident(ident) => convert_ident(ident),
        tt::Leaf::Punct(punct) => convert_punct(*punct),
    }
}

#[cfg(test)]
mod tests {
    use super::{convert_literal, TtToken};
    use parser::Token;
    use syntax::{SmolStr, SyntaxKind};

    #[test]
    fn test_negative_literal() {
        assert_eq!(
            convert_literal(&tt::Literal {
                id: tt::TokenId::unspecified(),
                text: SmolStr::new("-42.0")
            }),
            TtToken {
                tt: Token { kind: SyntaxKind::FLOAT_NUMBER, is_jointed_to_next: false },
                text: SmolStr::new("-42.0")
            }
        );
    }
}
