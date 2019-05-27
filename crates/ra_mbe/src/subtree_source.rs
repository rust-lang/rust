use ra_parser::{TokenSource, Token};
use ra_syntax::{classify_literal, SmolStr, SyntaxKind, SyntaxKind::*, T};
use std::cell::{RefCell, Cell};
use std::sync::Arc;
use tt::buffer::{TokenBuffer, Cursor};

pub(crate) trait Querier {
    fn token(&self, uidx: usize) -> (SyntaxKind, SmolStr, bool);
}

#[derive(Debug, Clone, Eq, PartialEq)]
struct TtToken {
    pub kind: SyntaxKind,
    pub is_joint_to_next: bool,
    pub text: SmolStr,
}

// A wrapper class for ref cell
#[derive(Debug)]
pub(crate) struct SubtreeWalk<'a> {
    start: Cursor<'a>,
    cursor: Cell<Cursor<'a>>,
    cached: RefCell<Vec<Option<TtToken>>>,
}

impl<'a> SubtreeWalk<'a> {
    fn new(cursor: Cursor<'a>) -> Self {
        SubtreeWalk {
            start: cursor,
            cursor: Cell::new(cursor),
            cached: RefCell::new(Vec::with_capacity(10)),
        }
    }

    fn get(&self, pos: usize) -> Option<TtToken> {
        let mut cached = self.cached.borrow_mut();
        if pos < cached.len() {
            return cached[pos].clone();
        }

        while pos >= cached.len() {
            let cursor = self.cursor.get();
            if cursor.eof() {
                cached.push(None);
                continue;
            }

            match cursor.token_tree() {
                Some(tt::TokenTree::Leaf(leaf)) => {
                    cached.push(Some(convert_leaf(&leaf)));
                    self.cursor.set(cursor.bump());
                }
                Some(tt::TokenTree::Subtree(subtree)) => {
                    self.cursor.set(cursor.subtree().unwrap());
                    cached.push(Some(convert_delim(subtree.delimiter, false)));
                }
                None => {
                    if let Some(subtree) = cursor.end() {
                        cached.push(Some(convert_delim(subtree.delimiter, true)));
                        self.cursor.set(cursor.bump());
                    }
                }
            }
        }

        return cached[pos].clone();
    }

    fn collect_token_trees(&self, n: usize) -> Vec<tt::TokenTree> {
        let mut res = vec![];

        let mut pos = 0;
        let mut cursor = self.start;
        let mut level = 0;

        while pos < n {
            if cursor.eof() {
                break;
            }

            match cursor.token_tree() {
                Some(tt::TokenTree::Leaf(leaf)) => {
                    if level == 0 {
                        res.push(leaf.into());
                    }
                    cursor = cursor.bump();
                    pos += 1;
                }
                Some(tt::TokenTree::Subtree(subtree)) => {
                    if level == 0 {
                        res.push(subtree.into());
                    }
                    pos += 1;
                    level += 1;
                    cursor = cursor.subtree().unwrap();
                }

                None => {
                    if let Some(_) = cursor.end() {
                        level -= 1;
                        pos += 1;
                        cursor = cursor.bump();
                    }
                }
            }
        }

        res
    }
}

impl<'a> Querier for SubtreeWalk<'a> {
    fn token(&self, uidx: usize) -> (SyntaxKind, SmolStr, bool) {
        self.get(uidx)
            .map(|tkn| (tkn.kind, tkn.text, tkn.is_joint_to_next))
            .unwrap_or_else(|| (SyntaxKind::EOF, "".into(), false))
    }
}

pub(crate) struct SubtreeTokenSource<'a> {
    walker: Arc<SubtreeWalk<'a>>,
    curr: (Token, usize),
}

impl<'a> SubtreeTokenSource<'a> {
    pub fn new(buffer: &'a TokenBuffer) -> SubtreeTokenSource<'a> {
        let mut res = SubtreeTokenSource {
            walker: Arc::new(SubtreeWalk::new(buffer.begin())),
            curr: (Token { kind: EOF, is_jointed_to_next: false }, 0),
        };
        res.curr = (res.mk_token(0), 0);
        res
    }

    pub fn querier(&self) -> Arc<SubtreeWalk<'a>> {
        self.walker.clone()
    }

    pub(crate) fn bump_n(&mut self, parsed_tokens: usize) -> Vec<tt::TokenTree> {
        let res = self.walker.collect_token_trees(parsed_tokens);
        res
    }

    fn mk_token(&self, pos: usize) -> Token {
        match self.walker.get(pos) {
            Some(tt) => Token { kind: tt.kind, is_jointed_to_next: tt.is_joint_to_next },
            None => Token { kind: EOF, is_jointed_to_next: false },
        }
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

        self.curr = (self.mk_token(self.curr.1 + 1), self.curr.1 + 1)
    }

    /// Is the current token a specified keyword?
    fn is_keyword(&self, kw: &str) -> bool {
        match self.walker.get(self.curr.1) {
            Some(t) => t.text == *kw,
            _ => false,
        }
    }
}

fn convert_delim(d: tt::Delimiter, closing: bool) -> TtToken {
    let (kinds, texts) = match d {
        tt::Delimiter::Parenthesis => ([T!['('], T![')']], "()"),
        tt::Delimiter::Brace => ([T!['{'], T!['}']], "{}"),
        tt::Delimiter::Bracket => ([T!['['], T![']']], "[]"),
        tt::Delimiter::None => ([L_DOLLAR, R_DOLLAR], ""),
    };

    let idx = closing as usize;
    let kind = kinds[idx];
    let text = if texts.len() > 0 { &texts[idx..texts.len() - (1 - idx)] } else { "" };
    TtToken { kind, is_joint_to_next: false, text: SmolStr::new(text) }
}

fn convert_literal(l: &tt::Literal) -> TtToken {
    let kind =
        classify_literal(&l.text).map(|tkn| tkn.kind).unwrap_or_else(|| match l.text.as_ref() {
            "true" => T![true],
            "false" => T![false],
            _ => panic!("Fail to convert given literal {:#?}", &l),
        });

    TtToken { kind, is_joint_to_next: false, text: l.text.clone() }
}

fn convert_ident(ident: &tt::Ident) -> TtToken {
    let kind = if let Some('\'') = ident.text.chars().next() {
        LIFETIME
    } else {
        SyntaxKind::from_keyword(ident.text.as_str()).unwrap_or(IDENT)
    };

    TtToken { kind, is_joint_to_next: false, text: ident.text.clone() }
}

fn convert_punct(p: &tt::Punct) -> TtToken {
    let kind = match p.char {
        // lexer may produce compound tokens for these ones
        '.' => T![.],
        ':' => T![:],
        '=' => T![=],
        '!' => T![!],
        '-' => T![-],
        c => SyntaxKind::from_char(c).unwrap(),
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
        tt::Leaf::Punct(punct) => convert_punct(punct),
    }
}
