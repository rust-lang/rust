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
enum WalkCursor {
    DelimiterBegin(Option<TtToken>),
    Token(usize, Option<TtToken>),
    DelimiterEnd(Option<TtToken>),
    Eof,
}

#[derive(Debug)]
struct SubTreeWalker<'a> {
    pos: usize,
    stack: Vec<(&'a tt::Subtree, Option<usize>)>,
    cursor: WalkCursor,
    last_steps: Vec<usize>,
    subtree: &'a tt::Subtree,
}

impl<'a> SubTreeWalker<'a> {
    fn new(subtree: &tt::Subtree) -> SubTreeWalker {
        let mut res = SubTreeWalker {
            pos: 0,
            stack: vec![],
            cursor: WalkCursor::Eof,
            last_steps: vec![],
            subtree,
        };

        res.reset();
        res
    }

    fn is_eof(&self) -> bool {
        self.cursor == WalkCursor::Eof
    }

    fn reset(&mut self) {
        self.pos = 0;
        self.stack = vec![(self.subtree, None)];
        self.cursor = WalkCursor::DelimiterBegin(convert_delim(self.subtree.delimiter, false));
        self.last_steps = vec![];

        while self.is_empty_delimiter() {
            self.forward_unchecked();
        }
    }

    // This funciton will fast forward the cursor,
    // Such that backward will stop at `start_pos` point
    fn start_from_nth(&mut self, start_pos: usize) {
        self.reset();
        self.pos = start_pos;
        self.cursor = self.walk_token(start_pos, 0, false);

        while self.is_empty_delimiter() {
            self.forward_unchecked();
        }
    }

    fn current(&self) -> Option<&TtToken> {
        match &self.cursor {
            WalkCursor::DelimiterBegin(t) => t.as_ref(),
            WalkCursor::Token(_, t) => t.as_ref(),
            WalkCursor::DelimiterEnd(t) => t.as_ref(),
            WalkCursor::Eof => None,
        }
    }

    fn is_empty_delimiter(&self) -> bool {
        match &self.cursor {
            WalkCursor::DelimiterBegin(None) => true,
            WalkCursor::DelimiterEnd(None) => true,
            _ => false,
        }
    }

    /// Move cursor backward by 1 step with empty checking
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

        // Move forward if it is empty delimiter
        if self.last_steps.is_empty() {
            while self.is_empty_delimiter() {
                self.forward_unchecked();
            }
        }
    }

    /// Move cursor backward by 1 step without empty check
    ///
    /// Depends on the current state of cursor:
    ///
    /// * Delimiter Begin => Pop the stack, goto last walking token  (`walk_token`)
    /// * Token => Goto prev token  (`walk_token`)
    /// * Delimiter End => Goto the last child token (`walk_token`)
    /// * Eof => push the root subtree, and set it as Delimiter End
    fn backward_unchecked(&mut self) {
        if self.last_steps.is_empty() {
            return;
        }

        let last_step = self.last_steps.pop().unwrap();
        let do_walk_token = match self.cursor {
            WalkCursor::DelimiterBegin(_) => None,
            WalkCursor::Token(u, _) => Some(u),
            WalkCursor::DelimiterEnd(_) => {
                let (top, _) = self.stack.last().unwrap();
                Some(top.token_trees.len())
            }
            WalkCursor::Eof => None,
        };

        self.cursor = match do_walk_token {
            Some(u) => self.walk_token(u, last_step, true),
            None => match self.cursor {
                WalkCursor::Eof => {
                    self.stack.push((self.subtree, None));
                    WalkCursor::DelimiterEnd(convert_delim(
                        self.stack.last().unwrap().0.delimiter,
                        true,
                    ))
                }
                _ => {
                    let (_, last_top_cursor) = self.stack.pop().unwrap();
                    assert!(!self.stack.is_empty());

                    self.walk_token(last_top_cursor.unwrap(), last_step, true)
                }
            },
        };
    }

    /// Move cursor forward by 1 step with empty checking
    fn forward(&mut self) {
        if self.is_eof() {
            return;
        }

        self.pos += 1;
        loop {
            self.forward_unchecked();
            if !self.is_empty_delimiter() {
                break;
            }
        }
    }

    /// Move cursor forward by 1 step without empty checking
    ///
    /// Depends on the current state of cursor:
    ///
    /// * Delimiter Begin => Goto the first child token (`walk_token`)
    /// * Token => Goto next token  (`walk_token`)
    /// * Delimiter End => Pop the stack, goto last walking token  (`walk_token`)
    ///   
    fn forward_unchecked(&mut self) {
        if self.is_eof() {
            return;
        }

        let step = self.current().map(|x| x.n_tokens).unwrap_or(1);
        self.last_steps.push(step);

        let do_walk_token = match self.cursor {
            WalkCursor::DelimiterBegin(_) => Some((0, 0)),
            WalkCursor::Token(u, _) => Some((u, step)),
            WalkCursor::DelimiterEnd(_) => None,
            _ => unreachable!(),
        };

        self.cursor = match do_walk_token {
            Some((u, step)) => self.walk_token(u, step, false),
            None => {
                let (_, last_top_idx) = self.stack.pop().unwrap();
                match self.stack.last() {
                    Some(_) => self.walk_token(last_top_idx.unwrap(), 1, false),
                    None => WalkCursor::Eof,
                }
            }
        };
    }

    /// Traversal child token
    /// Depends on the new position, it returns:
    ///
    /// * new position < 0 => DelimiterBegin
    /// * new position > token_tree.len() => DelimiterEnd
    /// * if new position is a subtree, depends on traversal direction:
    /// ** backward => DelimiterEnd
    /// ** forward => DelimiterBegin
    /// * if new psoition is a leaf, return walk_leaf()
    fn walk_token(&mut self, pos: usize, offset: usize, backward: bool) -> WalkCursor {
        let (top, _) = self.stack.last().unwrap();

        if backward && pos < offset {
            return WalkCursor::DelimiterBegin(convert_delim(
                self.stack.last().unwrap().0.delimiter,
                false,
            ));
        }

        if !backward && pos + offset >= top.token_trees.len() {
            return WalkCursor::DelimiterEnd(convert_delim(
                self.stack.last().unwrap().0.delimiter,
                true,
            ));
        }

        let pos = if backward { pos - offset } else { pos + offset };

        match &top.token_trees[pos] {
            tt::TokenTree::Subtree(subtree) => {
                self.stack.push((subtree, Some(pos)));
                let delim = convert_delim(self.stack.last().unwrap().0.delimiter, backward);
                if backward {
                    WalkCursor::DelimiterEnd(delim)
                } else {
                    WalkCursor::DelimiterBegin(delim)
                }
            }
            tt::TokenTree::Leaf(leaf) => WalkCursor::Token(pos, Some(self.walk_leaf(leaf, pos))),
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
#[derive(Debug)]
pub(crate) struct WalkerOwner<'a> {
    walker: RefCell<SubTreeWalker<'a>>,
    offset: usize,
}

impl<'a> WalkerOwner<'a> {
    fn new(subtree: &'a tt::Subtree) -> Self {
        WalkerOwner { walker: RefCell::new(SubTreeWalker::new(subtree)), offset: 0 }
    }

    fn get<'b>(&self, pos: usize) -> Option<TtToken> {
        self.set_walker_pos(pos);
        let walker = self.walker.borrow();
        walker.current().cloned()
    }

    fn start_from_nth(&mut self, pos: usize) {
        self.offset = pos;
        self.walker.borrow_mut().start_from_nth(pos);
    }

    fn set_walker_pos(&self, mut pos: usize) {
        pos += self.offset;
        let mut walker = self.walker.borrow_mut();
        while pos > walker.pos && !walker.is_eof() {
            walker.forward();
        }
        while pos < walker.pos {
            walker.backward();
        }
    }

    fn collect_token_trees(&mut self, n: usize) -> Vec<&tt::TokenTree> {
        self.start_from_nth(self.offset);

        let mut res = vec![];
        let mut walker = self.walker.borrow_mut();

        while walker.pos - self.offset < n {
            if let WalkCursor::Token(u, tt) = &walker.cursor {
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
        let tkn = self.get(uidx).unwrap();
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
        self.walker.get(pos).unwrap().is_joint_to_next
    }
    fn is_keyword(&self, pos: usize, kw: &str) -> bool {
        self.walker.get(pos).unwrap().text == *kw
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
    let iter = parent.token_trees[next + 1..].iter();
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
