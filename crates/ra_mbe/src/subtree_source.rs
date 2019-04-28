use ra_parser::{TokenSource};
use ra_syntax::{classify_literal, SmolStr, SyntaxKind, SyntaxKind::*};
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

    fn len(&self) -> usize {
        match self {
            TokenSeq::Subtree(subtree) => subtree.token_trees.len() + 2,
            TokenSeq::Seq(tokens) => tokens.len(),
        }
    }

    fn child_slice(&self, pos: usize) -> &[tt::TokenTree] {
        match self {
            TokenSeq::Subtree(subtree) => &subtree.token_trees[pos - 1..],
            TokenSeq::Seq(tokens) => &tokens[pos..],
        }
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
struct TtToken {
    pub kind: SyntaxKind,
    pub is_joint_to_next: bool,
    pub text: SmolStr,
    pub n_tokens: usize,
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
    last_steps: Vec<usize>,
    ts: TokenSeq<'a>,
}

impl<'a> SubTreeWalker<'a> {
    fn new(ts: TokenSeq<'a>) -> SubTreeWalker {
        let mut res = SubTreeWalker {
            pos: 0,
            stack: vec![],
            cursor: WalkCursor::Eof,
            last_steps: vec![],
            ts,
        };

        res.reset();
        res
    }

    fn is_eof(&self) -> bool {
        self.cursor == WalkCursor::Eof
    }

    fn reset(&mut self) {
        self.pos = 0;
        self.stack = vec![];
        self.last_steps = vec![];

        self.cursor = match self.ts.get(0) {
            DelimToken::Token(token) => match token {
                tt::TokenTree::Subtree(subtree) => {
                    let ts = TokenSeq::from(subtree);
                    self.stack.push((ts, 0));
                    WalkCursor::Token(0, convert_delim(subtree.delimiter, false))
                }
                tt::TokenTree::Leaf(leaf) => {
                    let next_tokens = self.ts.child_slice(0);
                    WalkCursor::Token(0, convert_leaf(&next_tokens, leaf))
                }
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

    /// Move cursor backward by 1 step
    fn backward(&mut self) {
        if self.last_steps.is_empty() {
            return;
        }

        self.pos -= 1;
        let last_step = self.last_steps.pop().unwrap();

        self.cursor = match self.cursor {
            WalkCursor::Token(idx, _) => self.walk_token(idx, last_step, true),
            WalkCursor::Eof => {
                let len = self.top().len();
                self.walk_token(len, last_step, true)
            }
        }
    }

    /// Move cursor forward by 1 step        
    fn forward(&mut self) {
        if self.is_eof() {
            return;
        }
        self.pos += 1;

        let step = self.current().map(|x| x.n_tokens).unwrap_or(1);
        self.last_steps.push(step);

        if let WalkCursor::Token(u, _) = self.cursor {
            self.cursor = self.walk_token(u, step, false)
        }
    }

    /// Traversal child token
    fn walk_token(&mut self, pos: usize, offset: usize, backward: bool) -> WalkCursor {
        let top = self.stack.last().map(|(t, _)| t).unwrap_or(&self.ts);

        if backward && pos < offset {
            let (_, last_idx) = self.stack.pop().unwrap();
            return self.walk_token(last_idx, offset, backward);
        }

        let pos = if backward { pos - offset } else { pos + offset };

        match top.get(pos) {
            DelimToken::Token(token) => match token {
                tt::TokenTree::Subtree(subtree) => {
                    let ts = TokenSeq::from(subtree);
                    let new_idx = if backward { ts.len() - 1 } else { 0 };
                    self.stack.push((ts, pos));
                    WalkCursor::Token(new_idx, convert_delim(subtree.delimiter, backward))
                }
                tt::TokenTree::Leaf(leaf) => {
                    let next_tokens = top.child_slice(pos);
                    WalkCursor::Token(pos, convert_leaf(&next_tokens, leaf))
                }
            },
            DelimToken::Delim(delim, is_end) => {
                WalkCursor::Token(pos, convert_delim(*delim, is_end))
            }
            DelimToken::End => {
                // it is the top level
                if let Some((_, last_idx)) = self.stack.pop() {
                    assert!(!backward);
                    self.walk_token(last_idx, offset, backward)
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
            let len = cached.len();
            cached.push({
                self.set_pos(len);
                let walker = self.walker.borrow();
                walker.current().cloned()
            });
        }

        return cached[pos].clone();
    }

    fn set_pos(&self, pos: usize) {
        let mut walker = self.walker.borrow_mut();
        while pos > walker.pos && !walker.is_eof() {
            walker.forward();
        }
        while pos < walker.pos {
            walker.backward();
        }
    }

    fn collect_token_trees(&mut self, n: usize) -> Vec<&tt::TokenTree> {
        let mut res = vec![];
        let mut walker = self.walker.borrow_mut();
        walker.reset();

        while walker.pos < n {
            if let WalkCursor::Token(u, tt) = &walker.cursor {
                // We only collect the topmost child
                if walker.stack.len() == 0 {
                    for i in 0..tt.n_tokens {
                        if let DelimToken::Token(token) = walker.ts.get(u + i) {
                            res.push(token);
                        }
                    }
                } else if walker.stack.len() == 1 {
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

pub(crate) struct TokenPeek<'a, I>
where
    I: Iterator<Item = &'a tt::TokenTree>,
{
    iter: itertools::MultiPeek<I>,
}

// helper function
fn to_punct(tt: &tt::TokenTree) -> Option<&tt::Punct> {
    if let tt::TokenTree::Leaf(tt::Leaf::Punct(pp)) = tt {
        return Some(pp);
    }
    None
}

impl<'a, I> TokenPeek<'a, I>
where
    I: Iterator<Item = &'a tt::TokenTree>,
{
    pub fn new(iter: I) -> Self {
        TokenPeek { iter: itertools::multipeek(iter) }
    }

    pub fn current_punct2(&mut self, p: &tt::Punct) -> Option<((char, char), bool)> {
        if p.spacing != tt::Spacing::Joint {
            return None;
        }

        self.iter.reset_peek();
        let p1 = to_punct(self.iter.peek()?)?;
        Some(((p.char, p1.char), p1.spacing == tt::Spacing::Joint))
    }

    pub fn current_punct3(&mut self, p: &tt::Punct) -> Option<((char, char, char), bool)> {
        self.current_punct2(p).and_then(|((p0, p1), last_joint)| {
            if !last_joint {
                None
            } else {
                let p2 = to_punct(*self.iter.peek()?)?;
                Some(((p0, p1, p2.char), p2.spacing == tt::Spacing::Joint))
            }
        })
    }
}

// FIXME: Remove this function
fn convert_multi_char_punct<'b, I>(
    p: &tt::Punct,
    iter: &mut TokenPeek<'b, I>,
) -> Option<(SyntaxKind, bool, &'static str, usize)>
where
    I: Iterator<Item = &'b tt::TokenTree>,
{
    if let Some((m, is_joint_to_next)) = iter.current_punct3(p) {
        if let Some((kind, text)) = match m {
            _ => None,
        } {
            return Some((kind, is_joint_to_next, text, 3));
        }
    }

    if let Some((m, is_joint_to_next)) = iter.current_punct2(p) {
        if let Some((kind, text)) = match m {
            _ => None,
        } {
            return Some((kind, is_joint_to_next, text, 2));
        }
    }

    None
}

fn convert_delim(d: tt::Delimiter, closing: bool) -> TtToken {
    let (kinds, texts) = match d {
        tt::Delimiter::Parenthesis => ([L_PAREN, R_PAREN], "()"),
        tt::Delimiter::Brace => ([L_CURLY, R_CURLY], "{}"),
        tt::Delimiter::Bracket => ([L_BRACK, R_BRACK], "[]"),
        tt::Delimiter::None => ([L_DOLLAR, R_DOLLAR], ""),
    };

    let idx = closing as usize;
    let kind = kinds[idx];
    let text = if texts.len() > 0 { &texts[idx..texts.len() - (1 - idx)] } else { "" };
    TtToken { kind, is_joint_to_next: false, text: SmolStr::new(text), n_tokens: 1 }
}

fn convert_literal(l: &tt::Literal) -> TtToken {
    let kind =
        classify_literal(&l.text).map(|tkn| tkn.kind).unwrap_or_else(|| match l.text.as_ref() {
            "true" => SyntaxKind::TRUE_KW,
            "false" => SyntaxKind::FALSE_KW,
            _ => panic!("Fail to convert given literal {:#?}", &l),
        });

    TtToken { kind, is_joint_to_next: false, text: l.text.clone(), n_tokens: 1 }
}

fn convert_ident(ident: &tt::Ident) -> TtToken {
    let kind = if let Some('\'') = ident.text.chars().next() {
        LIFETIME
    } else {
        SyntaxKind::from_keyword(ident.text.as_str()).unwrap_or(IDENT)
    };

    TtToken { kind, is_joint_to_next: false, text: ident.text.clone(), n_tokens: 1 }
}

fn convert_punct(p: &tt::Punct, next_tokens: &[tt::TokenTree]) -> TtToken {
    let mut iter = next_tokens.iter();
    iter.next();
    let mut peek = TokenPeek::new(iter);

    if let Some((kind, is_joint_to_next, text, size)) = convert_multi_char_punct(p, &mut peek) {
        TtToken { kind, is_joint_to_next, text: text.into(), n_tokens: size }
    } else {
        let kind = match p.char {
            // lexer may produce combpund tokens for these ones
            '.' => DOT,
            ':' => COLON,
            '=' => EQ,
            '!' => EXCL,
            '-' => MINUS,
            c => SyntaxKind::from_char(c).unwrap(),
        };
        let text = {
            let mut buf = [0u8; 4];
            let s: &str = p.char.encode_utf8(&mut buf);
            SmolStr::new(s)
        };
        TtToken { kind, is_joint_to_next: p.spacing == tt::Spacing::Joint, text, n_tokens: 1 }
    }
}

fn convert_leaf(tokens: &[tt::TokenTree], leaf: &tt::Leaf) -> TtToken {
    match leaf {
        tt::Leaf::Literal(l) => convert_literal(l),
        tt::Leaf::Ident(ident) => convert_ident(ident),
        tt::Leaf::Punct(punct) => convert_punct(punct, tokens),
    }
}
