//! FIXME: write short doc here

use parser::{Token, TokenSource};
use std::cell::{Cell, Ref, RefCell};
use syntax::{lex_single_syntax_kind, SmolStr, SyntaxKind, SyntaxKind::*, T};
use tt::buffer::{Cursor, TokenBuffer};

#[derive(Debug, Clone, Eq, PartialEq)]
struct TtToken {
    kind: SyntaxKind,
    is_joint_to_next: bool,
    text: SmolStr,
}

pub(crate) struct SubtreeTokenSource<'a> {
    cached_cursor: Cell<Cursor<'a>>,
    cached: RefCell<Vec<Option<TtToken>>>,
    curr: (Token, usize),
}

impl<'a> SubtreeTokenSource<'a> {
    // Helper function used in test
    #[cfg(test)]
    pub(crate) fn text(&self) -> SmolStr {
        match *self.get(self.curr.1) {
            Some(ref tt) => tt.text.clone(),
            _ => SmolStr::new(""),
        }
    }
}

impl<'a> SubtreeTokenSource<'a> {
    pub(crate) fn new(buffer: &'a TokenBuffer) -> SubtreeTokenSource<'a> {
        let cursor = buffer.begin();

        let mut res = SubtreeTokenSource {
            curr: (Token { kind: EOF, is_jointed_to_next: false }, 0),
            cached_cursor: Cell::new(cursor),
            cached: RefCell::new(Vec::with_capacity(10)),
        };
        res.curr = (res.mk_token(0), 0);
        res
    }

    fn mk_token(&self, pos: usize) -> Token {
        match *self.get(pos) {
            Some(ref tt) => Token { kind: tt.kind, is_jointed_to_next: tt.is_joint_to_next },
            None => Token { kind: EOF, is_jointed_to_next: false },
        }
    }

    fn get(&self, pos: usize) -> Ref<Option<TtToken>> {
        fn is_lifetime(c: Cursor) -> Option<(Cursor, SmolStr)> {
            let tkn = c.token_tree();

            if let Some(tt::buffer::TokenTreeRef::Leaf(tt::Leaf::Punct(punct), _)) = tkn {
                if punct.char == '\'' {
                    let next = c.bump();
                    if let Some(tt::buffer::TokenTreeRef::Leaf(tt::Leaf::Ident(ident), _)) =
                        next.token_tree()
                    {
                        let res_cursor = next.bump();
                        let text = SmolStr::new("'".to_string() + &ident.to_string());

                        return Some((res_cursor, text));
                    } else {
                        panic!("Next token must be ident : {:#?}", next.token_tree());
                    }
                }
            }

            None
        }

        if pos < self.cached.borrow().len() {
            return Ref::map(self.cached.borrow(), |c| &c[pos]);
        }

        {
            let mut cached = self.cached.borrow_mut();
            while pos >= cached.len() {
                let cursor = self.cached_cursor.get();
                if cursor.eof() {
                    cached.push(None);
                    continue;
                }

                if let Some((curr, text)) = is_lifetime(cursor) {
                    cached.push(Some(TtToken {
                        kind: LIFETIME_IDENT,
                        is_joint_to_next: false,
                        text,
                    }));
                    self.cached_cursor.set(curr);
                    continue;
                }

                match cursor.token_tree() {
                    Some(tt::buffer::TokenTreeRef::Leaf(leaf, _)) => {
                        cached.push(Some(convert_leaf(&leaf)));
                        self.cached_cursor.set(cursor.bump());
                    }
                    Some(tt::buffer::TokenTreeRef::Subtree(subtree, _)) => {
                        self.cached_cursor.set(cursor.subtree().unwrap());
                        cached.push(Some(convert_delim(subtree.delimiter_kind(), false)));
                    }
                    None => {
                        if let Some(subtree) = cursor.end() {
                            cached.push(Some(convert_delim(subtree.delimiter_kind(), true)));
                            self.cached_cursor.set(cursor.bump());
                        }
                    }
                }
            }
        }

        Ref::map(self.cached.borrow(), |c| &c[pos])
    }
}

impl<'a> TokenSource for SubtreeTokenSource<'a> {
    fn current(&self) -> Token {
        self.curr.0
    }

    /// Lookahead n token
    fn lookahead_nth(&self, n: usize) -> Token {
        self.mk_token(self.curr.1 + n)
    }

    /// bump cursor to next token
    fn bump(&mut self) {
        if self.current().kind == EOF {
            return;
        }

        self.curr = (self.mk_token(self.curr.1 + 1), self.curr.1 + 1);
    }

    /// Is the current token a specified keyword?
    fn is_keyword(&self, kw: &str) -> bool {
        match *self.get(self.curr.1) {
            Some(ref t) => t.text == *kw,
            _ => false,
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
    TtToken { kind, is_joint_to_next: false, text: SmolStr::new(text) }
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

    TtToken { kind, is_joint_to_next: false, text: l.text.clone() }
}

fn convert_ident(ident: &tt::Ident) -> TtToken {
    let kind = match ident.text.as_ref() {
        "true" => T![true],
        "false" => T![false],
        i if i.starts_with('\'') => LIFETIME_IDENT,
        _ => SyntaxKind::from_keyword(ident.text.as_str()).unwrap_or(IDENT),
    };

    TtToken { kind, is_joint_to_next: false, text: ident.text.clone() }
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
    TtToken { kind, is_joint_to_next: p.spacing == tt::Spacing::Joint, text }
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
    use syntax::{SmolStr, SyntaxKind};

    #[test]
    fn test_negative_literal() {
        assert_eq!(
            convert_literal(&tt::Literal {
                id: tt::TokenId::unspecified(),
                text: SmolStr::new("-42.0")
            }),
            TtToken {
                kind: SyntaxKind::FLOAT_NUMBER,
                is_joint_to_next: false,
                text: SmolStr::new("-42.0")
            }
        );
    }
}
