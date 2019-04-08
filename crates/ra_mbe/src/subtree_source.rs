use ra_parser::{TokenSource};
use ra_syntax::{classify_literal, SmolStr, SyntaxKind, SyntaxKind::*};
use std::cell::{RefCell};

#[derive(Debug, Clone, Eq, PartialEq)]
struct TtToken {
    pub kind: SyntaxKind,
    pub is_joint_to_next: bool,
    pub text: SmolStr,
    pub n_tokens: usize,
}

#[derive(Debug, Clone, Eq, PartialEq)]
enum WalkIndex {
    DelimiterBegin(Option<TtToken>),
    Token(usize, Option<TtToken>),
    DelimiterEnd(Option<TtToken>),
    Eof,
}

impl<'a> SubTreeWalker<'a> {
    fn new(subtree: &tt::Subtree) -> SubTreeWalker {
        let mut res = SubTreeWalker {
            pos: 0,
            stack: vec![],
            idx: WalkIndex::Eof,
            last_steps: vec![],
            subtree,
        };

        res.reset();
        res
    }

    fn reset(&mut self) {
        self.pos = 0;
        self.stack = vec![(self.subtree, None)];
        self.idx = WalkIndex::DelimiterBegin(convert_delim(self.subtree.delimiter, false));
        self.last_steps = vec![];

        while self.is_empty_delimiter() {
            self.forward_unchecked();
        }
    }

    // This funciton will fast forward the pos cursor,
    // Such that backward will stop at `start_pos` point
    fn start_from_nth(&mut self, start_pos: usize) {
        self.reset();
        self.pos = start_pos;
        self.idx = self.walk_token(start_pos, false);

        while self.is_empty_delimiter() {
            self.forward_unchecked();
        }
    }

    fn current(&self) -> Option<&TtToken> {
        match &self.idx {
            WalkIndex::DelimiterBegin(t) => t.as_ref(),
            WalkIndex::Token(_, t) => t.as_ref(),
            WalkIndex::DelimiterEnd(t) => t.as_ref(),
            WalkIndex::Eof => None,
        }
    }

    fn is_empty_delimiter(&self) -> bool {
        match &self.idx {
            WalkIndex::DelimiterBegin(None) => true,
            WalkIndex::DelimiterEnd(None) => true,
            _ => false,
        }
    }

    fn backward(&mut self) {
        if self.last_steps.is_empty() {
            return;
        }
        self.pos -= 1;
        loop {
            self.backward_unchecked();
            // Skip Empty delimiter
            if self.last_steps.is_empty() || !self.is_empty_delimiter() {
                break;
            }
        }
    }

    fn backward_unchecked(&mut self) {
        if self.last_steps.is_empty() {
            return;
        }

        let last_step = self.last_steps.pop().unwrap();
        let do_walk_token = match self.idx {
            WalkIndex::DelimiterBegin(_) => None,
            WalkIndex::Token(u, _) => Some(u),
            WalkIndex::DelimiterEnd(_) => {
                let (top, _) = self.stack.last().unwrap();
                Some(top.token_trees.len())
            }
            WalkIndex::Eof => None,
        };

        self.idx = match do_walk_token {
            Some(u) if last_step > u => WalkIndex::DelimiterBegin(convert_delim(
                self.stack.last().unwrap().0.delimiter,
                false,
            )),
            Some(u) => self.walk_token(u - last_step, true),
            None => match self.idx {
                WalkIndex::Eof => {
                    self.stack.push((self.subtree, None));
                    WalkIndex::DelimiterEnd(convert_delim(
                        self.stack.last().unwrap().0.delimiter,
                        true,
                    ))
                }
                _ => {
                    let (_, last_top_idx) = self.stack.pop().unwrap();
                    assert!(!self.stack.is_empty());

                    match last_top_idx.unwrap() {
                        0 => WalkIndex::DelimiterBegin(convert_delim(
                            self.stack.last().unwrap().0.delimiter,
                            false,
                        )),
                        c => self.walk_token(c - 1, true),
                    }
                }
            },
        };
    }

    fn forward(&mut self) {
        self.pos += 1;
        loop {
            self.forward_unchecked();
            if !self.is_empty_delimiter() {
                break;
            }
        }
    }

    fn forward_unchecked(&mut self) {
        if self.idx == WalkIndex::Eof {
            return;
        }

        let step = self.current().map(|x| x.n_tokens).unwrap_or(1);
        self.last_steps.push(step);

        let do_walk_token = match self.idx {
            WalkIndex::DelimiterBegin(_) => Some(0),
            WalkIndex::Token(u, _) => Some(u + step),
            WalkIndex::DelimiterEnd(_) => None,
            _ => unreachable!(),
        };

        let (top, _) = self.stack.last().unwrap();

        self.idx = match do_walk_token {
            Some(u) if u >= top.token_trees.len() => {
                WalkIndex::DelimiterEnd(convert_delim(self.stack.last().unwrap().0.delimiter, true))
            }
            Some(u) => self.walk_token(u, false),
            None => {
                let (_, last_top_idx) = self.stack.pop().unwrap();
                match self.stack.last() {
                    Some(top) => match last_top_idx.unwrap() {
                        idx if idx + 1 >= top.0.token_trees.len() => {
                            WalkIndex::DelimiterEnd(convert_delim(top.0.delimiter, true))
                        }
                        idx => self.walk_token(idx + 1, false),
                    },

                    None => WalkIndex::Eof,
                }
            }
        };
    }

    fn walk_token(&mut self, pos: usize, backward: bool) -> WalkIndex {
        let (top, _) = self.stack.last().unwrap();
        match &top.token_trees[pos] {
            tt::TokenTree::Subtree(subtree) => {
                self.stack.push((subtree, Some(pos)));
                let delim = convert_delim(self.stack.last().unwrap().0.delimiter, backward);
                if backward {
                    WalkIndex::DelimiterEnd(delim)
                } else {
                    WalkIndex::DelimiterBegin(delim)
                }
            }
            tt::TokenTree::Leaf(leaf) => WalkIndex::Token(pos, Some(self.walk_leaf(leaf, pos))),
        }
    }

    fn walk_leaf(&mut self, leaf: &tt::Leaf, pos: usize) -> TtToken {
        match leaf {
            tt::Leaf::Literal(l) => convert_literal(l),
            tt::Leaf::Ident(ident) => convert_ident(ident),
            tt::Leaf::Punct(punct) => {
                let (top, _) = self.stack.last().unwrap();
                convert_punct(punct, top, pos)
            }
        }
    }
}

pub(crate) trait Querier {
    fn token(&self, uidx: usize) -> (SyntaxKind, SmolStr);
}

// A wrapper class for ref cell
pub(crate) struct WalkerOwner<'a> {
    walker: RefCell<SubTreeWalker<'a>>,
    offset: usize,
}

impl<'a> WalkerOwner<'a> {
    fn token_idx<'b>(&self, pos: usize) -> Option<TtToken> {
        self.set_walker_pos(pos);
        self.walker.borrow().current().cloned()
    }

    fn start_from_nth(&mut self, pos: usize) {
        self.offset = pos;
        self.walker.borrow_mut().start_from_nth(pos);
    }

    fn set_walker_pos(&self, mut pos: usize) {
        pos += self.offset;
        let mut walker = self.walker.borrow_mut();
        while pos > walker.pos {
            walker.forward();
        }
        while pos < walker.pos {
            walker.backward();
        }
        assert!(pos == walker.pos);
    }

    fn new(subtree: &'a tt::Subtree) -> Self {
        WalkerOwner { walker: RefCell::new(SubTreeWalker::new(subtree)), offset: 0 }
    }

    fn collect_token_tree(&mut self, n: usize) -> Vec<&tt::TokenTree> {
        self.start_from_nth(self.offset);

        let mut res = vec![];
        let mut walker = self.walker.borrow_mut();

        while walker.pos - self.offset < n {
            if let WalkIndex::Token(u, tt) = &walker.idx {
                if walker.stack.len() == 1 {
                    // We only collect the topmost child
                    res.push(&walker.stack[0].0.token_trees[*u]);
                    if let Some(tt) = tt {
                        for i in 0..tt.n_tokens - 1 {
                            res.push(&walker.stack[0].0.token_trees[u + i]);
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
    fn token(&self, uidx: usize) -> (SyntaxKind, SmolStr) {
        let tkn = self.token_idx(uidx).unwrap();
        (tkn.kind, tkn.text)
    }
}

pub(crate) struct SubtreeTokenSource<'a> {
    walker: WalkerOwner<'a>,
}

impl<'a> SubtreeTokenSource<'a> {
    pub fn new(subtree: &tt::Subtree) -> SubtreeTokenSource {
        SubtreeTokenSource { walker: WalkerOwner::new(subtree) }
    }

    pub fn start_from_nth(&mut self, n: usize) {
        self.walker.start_from_nth(n);
    }

    pub fn querier<'b>(&'a self) -> &'b WalkerOwner<'a>
    where
        'a: 'b,
    {
        &self.walker
    }

    pub(crate) fn bump_n(
        &mut self,
        parsed_tokens: usize,
        cursor_pos: &mut usize,
    ) -> Vec<&tt::TokenTree> {
        let res = self.walker.collect_token_tree(parsed_tokens);
        *cursor_pos += res.len();

        res
    }
}

impl<'a> TokenSource for SubtreeTokenSource<'a> {
    fn token_kind(&self, pos: usize) -> SyntaxKind {
        if let Some(tok) = self.walker.token_idx(pos) {
            tok.kind
        } else {
            SyntaxKind::EOF
        }
    }
    fn is_token_joint_to_next(&self, pos: usize) -> bool {
        self.walker.token_idx(pos).unwrap().is_joint_to_next
    }
    fn is_keyword(&self, pos: usize, kw: &str) -> bool {
        self.walker.token_idx(pos).unwrap().text == *kw
    }
}

struct TokenPeek<'a, I>
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

    fn current_punct2(&mut self, p: &tt::Punct) -> Option<((char, char), bool)> {
        if p.spacing != tt::Spacing::Joint {
            return None;
        }

        self.iter.reset_peek();
        let p1 = to_punct(self.iter.peek()?)?;
        Some(((p.char, p1.char), p1.spacing == tt::Spacing::Joint))
    }

    fn current_punct3(&mut self, p: &tt::Punct) -> Option<((char, char, char), bool)> {
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

fn convert_multi_char_punct<'b, I>(
    p: &tt::Punct,
    iter: &mut TokenPeek<'b, I>,
) -> Option<(SyntaxKind, bool, &'static str, usize)>
where
    I: Iterator<Item = &'b tt::TokenTree>,
{
    if let Some((m, is_joint_to_next)) = iter.current_punct3(p) {
        if let Some((kind, text)) = match m {
            ('<', '<', '=') => Some((SHLEQ, "<<=")),
            ('>', '>', '=') => Some((SHREQ, ">>=")),
            ('.', '.', '.') => Some((DOTDOTDOT, "...")),
            ('.', '.', '=') => Some((DOTDOTEQ, "..=")),
            _ => None,
        } {
            return Some((kind, is_joint_to_next, text, 3));
        }
    }

    if let Some((m, is_joint_to_next)) = iter.current_punct2(p) {
        if let Some((kind, text)) = match m {
            ('<', '<') => Some((SHL, "<<")),
            ('>', '>') => Some((SHR, ">>")),

            ('|', '|') => Some((PIPEPIPE, "||")),
            ('&', '&') => Some((AMPAMP, "&&")),
            ('%', '=') => Some((PERCENTEQ, "%=")),
            ('*', '=') => Some((STAREQ, "*=")),
            ('/', '=') => Some((SLASHEQ, "/=")),
            ('^', '=') => Some((CARETEQ, "^=")),

            ('&', '=') => Some((AMPEQ, "&=")),
            ('|', '=') => Some((PIPEEQ, "|=")),
            ('-', '=') => Some((MINUSEQ, "-=")),
            ('+', '=') => Some((PLUSEQ, "+=")),
            ('>', '=') => Some((GTEQ, ">=")),
            ('<', '=') => Some((LTEQ, "<=")),

            ('-', '>') => Some((THIN_ARROW, "->")),
            ('!', '=') => Some((NEQ, "!=")),
            ('=', '>') => Some((FAT_ARROW, "=>")),
            ('=', '=') => Some((EQEQ, "==")),
            ('.', '.') => Some((DOTDOT, "..")),
            (':', ':') => Some((COLONCOLON, "::")),

            _ => None,
        } {
            return Some((kind, is_joint_to_next, text, 2));
        }
    }

    None
}

struct SubTreeWalker<'a> {
    pos: usize,
    stack: Vec<(&'a tt::Subtree, Option<usize>)>,
    idx: WalkIndex,
    last_steps: Vec<usize>,
    subtree: &'a tt::Subtree,
}

fn convert_delim(d: tt::Delimiter, closing: bool) -> Option<TtToken> {
    let (kinds, texts) = match d {
        tt::Delimiter::Parenthesis => ([L_PAREN, R_PAREN], "()"),
        tt::Delimiter::Brace => ([L_CURLY, R_CURLY], "{}"),
        tt::Delimiter::Bracket => ([L_BRACK, R_BRACK], "[]"),
        tt::Delimiter::None => return None,
    };

    let idx = closing as usize;
    let kind = kinds[idx];
    let text = &texts[idx..texts.len() - (1 - idx)];
    Some(TtToken { kind, is_joint_to_next: false, text: SmolStr::new(text), n_tokens: 1 })
}

fn convert_literal(l: &tt::Literal) -> TtToken {
    TtToken {
        kind: classify_literal(&l.text).unwrap().kind,
        is_joint_to_next: false,
        text: l.text.clone(),
        n_tokens: 1,
    }
}

fn convert_ident(ident: &tt::Ident) -> TtToken {
    let kind = SyntaxKind::from_keyword(ident.text.as_str()).unwrap_or(IDENT);
    TtToken { kind, is_joint_to_next: false, text: ident.text.clone(), n_tokens: 1 }
}

fn convert_punct(p: &tt::Punct, parent: &tt::Subtree, next: usize) -> TtToken {
    let iter = parent.token_trees[next..].iter();
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
