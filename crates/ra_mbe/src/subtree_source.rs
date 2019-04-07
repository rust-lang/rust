use ra_parser::{TokenSource};
use ra_syntax::{classify_literal, SmolStr, SyntaxKind, SyntaxKind::*};

#[derive(Debug)]
struct TtToken {
    pub kind: SyntaxKind,
    pub is_joint_to_next: bool,
    pub text: SmolStr,
    pub n_tokens: usize,
}

/// Querier let outside to query internal tokens as string
pub(crate) struct Querier<'a> {
    src: &'a SubtreeTokenSource<'a>,
}

impl<'a> Querier<'a> {
    pub(crate) fn token(&self, uidx: usize) -> (SyntaxKind, &SmolStr) {
        let tkn = &self.src.tokens[uidx];
        (tkn.kind, &tkn.text)
    }
}

pub(crate) struct SubtreeTokenSource<'a> {
    tt_pos: usize,
    tokens: Vec<TtToken>,
    subtree: &'a tt::Subtree,
}

impl<'a> SubtreeTokenSource<'a> {
    pub fn new(subtree: &tt::Subtree) -> SubtreeTokenSource {
        SubtreeTokenSource { tokens: TtTokenBuilder::build(subtree), tt_pos: 0, subtree }
    }

    // Advance token source and skip the first delimiter
    pub fn advance(&mut self, n_token: usize, skip_first_delimiter: bool) {
        if skip_first_delimiter {
            self.tt_pos += 1;
        }

        // Matching `TtToken` cursor to `tt::TokenTree` cursor
        // It is because TtToken is not One to One mapping to tt::Token
        // There are 3 case (`TtToken` <=> `tt::TokenTree`) :
        // * One to One =>  ident, single char punch
        // * Many to One => `tt::TokenTree::SubTree`
        // * One to Many => multibyte punct
        //
        // Such that we cannot simpliy advance the cursor
        // We have to bump it one by one
        let mut pos = 0;
        while pos < n_token {
            pos += self.bump(&self.subtree.token_trees[pos]);
        }
    }

    pub fn querier(&self) -> Querier {
        Querier { src: self }
    }

    pub(crate) fn bump_n(
        &mut self,
        n_tt_tokens: usize,
        token_pos: &mut usize,
    ) -> Vec<&tt::TokenTree> {
        let mut res = vec![];
        // Matching `TtToken` cursor to `tt::TokenTree` cursor
        // It is because TtToken is not One to One mapping to tt::Token
        // There are 3 case (`TtToken` <=> `tt::TokenTree`) :
        // * One to One =>  ident, single char punch
        // * Many to One => `tt::TokenTree::SubTree`
        // * One to Many => multibyte punct
        //
        // Such that we cannot simpliy advance the cursor
        // We have to bump it one by one
        let next_pos = self.tt_pos + n_tt_tokens;

        while self.tt_pos < next_pos {
            let current = &self.subtree.token_trees[*token_pos];
            let n = self.bump(current);
            res.extend((0..n).map(|i| &self.subtree.token_trees[*token_pos + i]));
            *token_pos += n;
        }

        res
    }

    fn count(&self, tt: &tt::TokenTree) -> usize {
        assert!(!self.tokens.is_empty());
        TtTokenBuilder::count_tt_tokens(tt, None)
    }

    fn bump(&mut self, tt: &tt::TokenTree) -> usize {
        let cur = &self.tokens[self.tt_pos];
        let n_tokens = cur.n_tokens;
        self.tt_pos += self.count(tt);
        n_tokens
    }
}

impl<'a> TokenSource for SubtreeTokenSource<'a> {
    fn token_kind(&self, pos: usize) -> SyntaxKind {
        if let Some(tok) = self.tokens.get(self.tt_pos + pos) {
            tok.kind
        } else {
            SyntaxKind::EOF
        }
    }
    fn is_token_joint_to_next(&self, pos: usize) -> bool {
        self.tokens[self.tt_pos + pos].is_joint_to_next
    }
    fn is_keyword(&self, pos: usize, kw: &str) -> bool {
        self.tokens[self.tt_pos + pos].text == *kw
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

    pub fn next(&mut self) -> Option<&tt::TokenTree> {
        self.iter.next()
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

struct TtTokenBuilder {
    tokens: Vec<TtToken>,
}

impl TtTokenBuilder {
    fn build(sub: &tt::Subtree) -> Vec<TtToken> {
        let mut res = TtTokenBuilder { tokens: vec![] };
        res.convert_subtree(sub);
        res.tokens
    }

    fn convert_subtree(&mut self, sub: &tt::Subtree) {
        self.push_delim(sub.delimiter, false);
        let mut peek = TokenPeek::new(sub.token_trees.iter());
        while let Some(tt) = peek.iter.next() {
            self.convert_tt(tt, &mut peek);
        }
        self.push_delim(sub.delimiter, true)
    }

    fn convert_tt<'b, I>(&mut self, tt: &tt::TokenTree, iter: &mut TokenPeek<'b, I>)
    where
        I: Iterator<Item = &'b tt::TokenTree>,
    {
        match tt {
            tt::TokenTree::Leaf(token) => self.convert_token(token, iter),
            tt::TokenTree::Subtree(sub) => self.convert_subtree(sub),
        }
    }

    fn convert_token<'b, I>(&mut self, token: &tt::Leaf, iter: &mut TokenPeek<'b, I>)
    where
        I: Iterator<Item = &'b tt::TokenTree>,
    {
        let tok = match token {
            tt::Leaf::Literal(l) => TtToken {
                kind: classify_literal(&l.text).unwrap().kind,
                is_joint_to_next: false,
                text: l.text.clone(),
                n_tokens: 1,
            },
            tt::Leaf::Punct(p) => {
                if let Some((kind, is_joint_to_next, text, size)) =
                    Self::convert_multi_char_punct(p, iter)
                {
                    for _ in 0..size - 1 {
                        iter.next();
                    }

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
                    TtToken {
                        kind,
                        is_joint_to_next: p.spacing == tt::Spacing::Joint,
                        text,
                        n_tokens: 1,
                    }
                }
            }
            tt::Leaf::Ident(ident) => {
                let kind = SyntaxKind::from_keyword(ident.text.as_str()).unwrap_or(IDENT);
                TtToken { kind, is_joint_to_next: false, text: ident.text.clone(), n_tokens: 1 }
            }
        };
        self.tokens.push(tok)
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

    fn push_delim(&mut self, d: tt::Delimiter, closing: bool) {
        let (kinds, texts) = match d {
            tt::Delimiter::Parenthesis => ([L_PAREN, R_PAREN], "()"),
            tt::Delimiter::Brace => ([L_CURLY, R_CURLY], "{}"),
            tt::Delimiter::Bracket => ([L_BRACK, R_BRACK], "[]"),
            tt::Delimiter::None => return,
        };
        let idx = closing as usize;
        let kind = kinds[idx];
        let text = &texts[idx..texts.len() - (1 - idx)];
        let tok = TtToken { kind, is_joint_to_next: false, text: SmolStr::new(text), n_tokens: 1 };
        self.tokens.push(tok)
    }

    fn skip_sibling_leaf(leaf: &tt::Leaf, iter: &mut std::slice::Iter<tt::TokenTree>) {
        if let tt::Leaf::Punct(p) = leaf {
            let mut peek = TokenPeek::new(iter);
            if let Some((_, _, _, size)) = TtTokenBuilder::convert_multi_char_punct(p, &mut peek) {
                for _ in 0..size - 1 {
                    peek.next();
                }
            }
        }
    }

    fn count_tt_tokens(
        tt: &tt::TokenTree,
        iter: Option<&mut std::slice::Iter<tt::TokenTree>>,
    ) -> usize {
        match tt {
            tt::TokenTree::Subtree(sub_tree) => {
                let mut iter = sub_tree.token_trees.iter();
                let mut count = match sub_tree.delimiter {
                    tt::Delimiter::None => 0,
                    _ => 2,
                };

                while let Some(tt) = iter.next() {
                    count += Self::count_tt_tokens(&tt, Some(&mut iter));
                }
                count
            }

            tt::TokenTree::Leaf(leaf) => {
                iter.map(|iter| {
                    Self::skip_sibling_leaf(leaf, iter);
                });

                1
            }
        }
    }
}
