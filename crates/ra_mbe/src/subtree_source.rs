use ra_parser::{TokenSource};
use ra_syntax::{classify_literal, SmolStr, SyntaxKind, SyntaxKind::*, T};
use std::cell::{RefCell};

// A Sequece of Token,
#[derive(Debug, Clone, Eq, PartialEq)]
pub(super) enum TokenSeq<'a> {
    Subtree(&'a tt::Subtree),
    Seq(&'a [tt::TokenTree]),
}

impl<'a> From<&'a tt::Subtree> for TokenSeq<'a> {
    fn from(s: &'a tt::Subtree) -> TokenSeq<'a> {
        TokenSeq::Subtree(s)
    }
}

impl<'a> From<&'a [tt::TokenTree]> for TokenSeq<'a> {
    fn from(s: &'a [tt::TokenTree]) -> TokenSeq<'a> {
        TokenSeq::Seq(s)
    }
}

#[derive(Debug)]
enum DelimToken<'a> {
    Delim(&'a tt::Delimiter, bool),
    Token(&'a tt::TokenTree),
    End,
}

impl<'a> TokenSeq<'a> {
    fn get(&self, pos: usize) -> DelimToken<'a> {
        match self {
            TokenSeq::Subtree(subtree) => {
                let len = subtree.token_trees.len() + 2;
                match pos {
                    p if p >= len => DelimToken::End,
                    p if p == len - 1 => DelimToken::Delim(&subtree.delimiter, true),
                    0 => DelimToken::Delim(&subtree.delimiter, false),
                    p => DelimToken::Token(&subtree.token_trees[p - 1]),
                }
            }
            TokenSeq::Seq(tokens) => {
                tokens.get(pos).map(DelimToken::Token).unwrap_or(DelimToken::End)
            }
        }
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
struct TtToken {
    pub kind: SyntaxKind,
    pub is_joint_to_next: bool,
    pub text: SmolStr,
}

#[derive(Debug, Clone, Eq, PartialEq)]
enum WalkCursor {
    Token(usize, TtToken),
    Eof,
}

#[derive(Debug)]
struct SubTreeWalker<'a> {
    pos: usize,
    stack: Vec<(TokenSeq<'a>, usize)>,
    cursor: WalkCursor,
    ts: TokenSeq<'a>,
}

impl<'a> SubTreeWalker<'a> {
    fn new(ts: TokenSeq<'a>) -> SubTreeWalker {
        let mut res = SubTreeWalker { pos: 0, stack: vec![], cursor: WalkCursor::Eof, ts };

        res.reset();
        res
    }

    fn is_eof(&self) -> bool {
        self.cursor == WalkCursor::Eof
    }

    fn reset(&mut self) {
        self.pos = 0;
        self.stack = vec![];

        self.cursor = match self.ts.get(0) {
            DelimToken::Token(token) => match token {
                tt::TokenTree::Subtree(subtree) => {
                    let ts = TokenSeq::from(subtree);
                    self.stack.push((ts, 0));
                    WalkCursor::Token(0, convert_delim(subtree.delimiter, false))
                }
                tt::TokenTree::Leaf(leaf) => WalkCursor::Token(0, convert_leaf(leaf)),
            },
            DelimToken::Delim(delim, is_end) => {
                assert!(!is_end);
                WalkCursor::Token(0, convert_delim(*delim, false))
            }
            DelimToken::End => WalkCursor::Eof,
        }
    }

    fn current(&self) -> Option<&TtToken> {
        match &self.cursor {
            WalkCursor::Token(_, t) => Some(t),
            WalkCursor::Eof => None,
        }
    }

    fn top(&self) -> &TokenSeq {
        self.stack.last().map(|(t, _)| t).unwrap_or(&self.ts)
    }

    /// Move cursor forward by 1 step        
    fn forward(&mut self) {
        if self.is_eof() {
            return;
        }
        self.pos += 1;

        if let WalkCursor::Token(u, _) = self.cursor {
            self.cursor = self.walk_token(u)
        }
    }

    /// Traversal child token
    fn walk_token(&mut self, pos: usize) -> WalkCursor {
        let top = self.stack.last().map(|(t, _)| t).unwrap_or(&self.ts);
        let pos = pos + 1;

        match top.get(pos) {
            DelimToken::Token(token) => match token {
                tt::TokenTree::Subtree(subtree) => {
                    let ts = TokenSeq::from(subtree);
                    self.stack.push((ts, pos));
                    WalkCursor::Token(0, convert_delim(subtree.delimiter, false))
                }
                tt::TokenTree::Leaf(leaf) => WalkCursor::Token(pos, convert_leaf(leaf)),
            },
            DelimToken::Delim(delim, is_end) => {
                WalkCursor::Token(pos, convert_delim(*delim, is_end))
            }
            DelimToken::End => {
                // it is the top level
                if let Some((_, last_idx)) = self.stack.pop() {
                    self.walk_token(last_idx)
                } else {
                    WalkCursor::Eof
                }
            }
        }
    }
}

pub(crate) trait Querier {
    fn token(&self, uidx: usize) -> (SyntaxKind, SmolStr, bool);
}

// A wrapper class for ref cell
#[derive(Debug)]
pub(crate) struct WalkerOwner<'a> {
    walker: RefCell<SubTreeWalker<'a>>,
    cached: RefCell<Vec<Option<TtToken>>>,
}

impl<'a> WalkerOwner<'a> {
    fn new<I: Into<TokenSeq<'a>>>(ts: I) -> Self {
        WalkerOwner {
            walker: RefCell::new(SubTreeWalker::new(ts.into())),
            cached: RefCell::new(Vec::with_capacity(10)),
        }
    }

    fn get<'b>(&self, pos: usize) -> Option<TtToken> {
        let mut cached = self.cached.borrow_mut();
        if pos < cached.len() {
            return cached[pos].clone();
        }

        while pos >= cached.len() {
            self.set_pos(cached.len());
            let walker = self.walker.borrow();
            cached.push(walker.current().cloned());
        }

        return cached[pos].clone();
    }

    fn set_pos(&self, pos: usize) {
        let mut walker = self.walker.borrow_mut();
        assert!(walker.pos <= pos);

        while pos > walker.pos && !walker.is_eof() {
            walker.forward();
        }
    }

    fn collect_token_trees(&mut self, n: usize) -> Vec<&tt::TokenTree> {
        let mut res = vec![];
        let mut walker = self.walker.borrow_mut();
        walker.reset();

        while walker.pos < n {
            if let WalkCursor::Token(u, _) = &walker.cursor {
                // We only collect the topmost child
                if walker.stack.len() == 0 {
                    if let DelimToken::Token(token) = walker.ts.get(*u) {
                        res.push(token);
                    }
                }
                // Check whether the second level is a subtree
                // if so, collect its parent which is topmost child
                else if walker.stack.len() == 1 {
                    if let DelimToken::Delim(_, is_end) = walker.top().get(*u) {
                        if !is_end {
                            let (_, last_idx) = &walker.stack[0];
                            if let DelimToken::Token(token) = walker.ts.get(*last_idx) {
                                res.push(token);
                            }
                        }
                    }
                }
            }

            walker.forward();
        }

        res
    }
}

impl<'a> Querier for WalkerOwner<'a> {
    fn token(&self, uidx: usize) -> (SyntaxKind, SmolStr, bool) {
        self.get(uidx)
            .map(|tkn| (tkn.kind, tkn.text, tkn.is_joint_to_next))
            .unwrap_or_else(|| (SyntaxKind::EOF, "".into(), false))
    }
}

pub(crate) struct SubtreeTokenSource<'a> {
    walker: WalkerOwner<'a>,
}

impl<'a> SubtreeTokenSource<'a> {
    pub fn new<I: Into<TokenSeq<'a>>>(ts: I) -> SubtreeTokenSource<'a> {
        SubtreeTokenSource { walker: WalkerOwner::new(ts) }
    }

    pub fn querier<'b>(&'a self) -> &'b WalkerOwner<'a>
    where
        'a: 'b,
    {
        &self.walker
    }

    pub(crate) fn bump_n(&mut self, parsed_tokens: usize) -> Vec<&tt::TokenTree> {
        let res = self.walker.collect_token_trees(parsed_tokens);
        res
    }
}

impl<'a> TokenSource for SubtreeTokenSource<'a> {
    fn token_kind(&self, pos: usize) -> SyntaxKind {
        if let Some(tok) = self.walker.get(pos) {
            tok.kind
        } else {
            SyntaxKind::EOF
        }
    }
    fn is_token_joint_to_next(&self, pos: usize) -> bool {
        match self.walker.get(pos) {
            Some(t) => t.is_joint_to_next,
            _ => false,
        }
    }
    fn is_keyword(&self, pos: usize, kw: &str) -> bool {
        match self.walker.get(pos) {
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
