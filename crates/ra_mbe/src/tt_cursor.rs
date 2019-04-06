use crate::ParseError;
use crate::syntax_bridge::{TtTokenSource, TtToken, TokenPeek};
use ra_parser::{TokenSource, TreeSink};

use ra_syntax::{
    SyntaxKind
};

struct TtCursorTokenSource {
    tt_pos: usize,
    inner: TtTokenSource,
}

impl TtCursorTokenSource {
    fn new(subtree: &tt::Subtree, curr: usize) -> TtCursorTokenSource {
        let mut res = TtCursorTokenSource { inner: TtTokenSource::new(subtree), tt_pos: 1 };

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
        while pos < curr {
            pos += res.bump(&subtree.token_trees[pos]);
        }

        res
    }

    fn skip_sibling_leaf(&self, leaf: &tt::Leaf, iter: &mut std::slice::Iter<tt::TokenTree>) {
        if let tt::Leaf::Punct(p) = leaf {
            let mut peek = TokenPeek::new(iter);
            if let Some((_, _, _, size)) = TtTokenSource::convert_multi_char_punct(p, &mut peek) {
                for _ in 0..size - 1 {
                    peek.next();
                }
            }
        }
    }

    fn count_tt_tokens(
        &self,
        tt: &tt::TokenTree,
        iter: Option<&mut std::slice::Iter<tt::TokenTree>>,
    ) -> usize {
        assert!(!self.inner.tokens.is_empty());

        match tt {
            tt::TokenTree::Subtree(sub_tree) => {
                let mut iter = sub_tree.token_trees.iter();
                let mut count = match sub_tree.delimiter {
                    tt::Delimiter::None => 0,
                    _ => 2,
                };

                while let Some(tt) = iter.next() {
                    count += self.count_tt_tokens(&tt, Some(&mut iter));
                }
                count
            }

            tt::TokenTree::Leaf(leaf) => {
                iter.map(|iter| {
                    self.skip_sibling_leaf(leaf, iter);
                });

                1
            }
        }
    }

    fn count(&self, tt: &tt::TokenTree) -> usize {
        self.count_tt_tokens(tt, None)
    }

    fn bump(&mut self, tt: &tt::TokenTree) -> usize {
        let cur = self.current().unwrap();
        let n_tokens = cur.n_tokens;
        self.tt_pos += self.count(tt);
        n_tokens
    }

    fn current(&self) -> Option<&TtToken> {
        self.inner.tokens.get(self.tt_pos)
    }
}

impl TokenSource for TtCursorTokenSource {
    fn token_kind(&self, pos: usize) -> SyntaxKind {
        if let Some(tok) = self.inner.tokens.get(self.tt_pos + pos) {
            tok.kind
        } else {
            SyntaxKind::EOF
        }
    }
    fn is_token_joint_to_next(&self, pos: usize) -> bool {
        self.inner.tokens[self.tt_pos + pos].is_joint_to_next
    }
    fn is_keyword(&self, pos: usize, kw: &str) -> bool {
        self.inner.tokens[self.tt_pos + pos].text == *kw
    }
}

struct TtCursorTokenSink {
    token_pos: usize,
}

impl TreeSink for TtCursorTokenSink {
    fn token(&mut self, _kind: SyntaxKind, n_tokens: u8) {
        self.token_pos += n_tokens as usize;
    }

    fn start_node(&mut self, _kind: SyntaxKind) {}
    fn finish_node(&mut self) {}
    fn error(&mut self, _error: ra_parser::ParseError) {}
}

#[derive(Clone)]
pub(crate) struct TtCursor<'a> {
    subtree: &'a tt::Subtree,
    pos: usize,
}

impl<'a> TtCursor<'a> {
    pub(crate) fn new(subtree: &'a tt::Subtree) -> TtCursor<'a> {
        TtCursor { subtree, pos: 0 }
    }

    pub(crate) fn is_eof(&self) -> bool {
        self.pos == self.subtree.token_trees.len()
    }

    pub(crate) fn current(&self) -> Option<&'a tt::TokenTree> {
        self.subtree.token_trees.get(self.pos)
    }

    pub(crate) fn at_punct(&self) -> Option<&'a tt::Punct> {
        match self.current() {
            Some(tt::TokenTree::Leaf(tt::Leaf::Punct(it))) => Some(it),
            _ => None,
        }
    }

    pub(crate) fn at_char(&self, char: char) -> bool {
        match self.at_punct() {
            Some(tt::Punct { char: c, .. }) if *c == char => true,
            _ => false,
        }
    }

    pub(crate) fn at_ident(&mut self) -> Option<&'a tt::Ident> {
        match self.current() {
            Some(tt::TokenTree::Leaf(tt::Leaf::Ident(i))) => Some(i),
            _ => None,
        }
    }

    pub(crate) fn bump(&mut self) {
        self.pos += 1;
    }
    pub(crate) fn rev_bump(&mut self) {
        self.pos -= 1;
    }

    pub(crate) fn eat(&mut self) -> Option<&'a tt::TokenTree> {
        self.current().map(|it| {
            self.bump();
            it
        })
    }

    pub(crate) fn eat_subtree(&mut self) -> Result<&'a tt::Subtree, ParseError> {
        match self.current() {
            Some(tt::TokenTree::Subtree(sub)) => {
                self.bump();
                Ok(sub)
            }
            _ => Err(ParseError::Expected(String::from("subtree"))),
        }
    }

    pub(crate) fn eat_punct(&mut self) -> Option<&'a tt::Punct> {
        self.at_punct().map(|it| {
            self.bump();
            it
        })
    }

    pub(crate) fn eat_ident(&mut self) -> Option<&'a tt::Ident> {
        self.at_ident().map(|i| {
            self.bump();
            i
        })
    }

    fn eat_parse_result(
        &mut self,
        parsed_token: usize,
        src: &mut TtCursorTokenSource,
    ) -> Option<tt::TokenTree> {
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
        let next_pos = src.tt_pos + parsed_token;
        while src.tt_pos < next_pos {
            let n = src.bump(self.current().unwrap());
            res.extend((0..n).map(|_| self.eat().unwrap()));
        }

        let res: Vec<_> = res.into_iter().cloned().collect();

        match res.len() {
            0 => None,
            1 => Some(res[0].clone()),
            _ => Some(tt::TokenTree::Subtree(tt::Subtree {
                delimiter: tt::Delimiter::None,
                token_trees: res,
            })),
        }
    }

    fn eat_parse<F>(&mut self, f: F) -> Option<tt::TokenTree>
    where
        F: FnOnce(&dyn TokenSource, &mut dyn TreeSink),
    {
        let mut src = TtCursorTokenSource::new(self.subtree, self.pos);
        let mut sink = TtCursorTokenSink { token_pos: 0 };

        f(&src, &mut sink);

        self.eat_parse_result(sink.token_pos, &mut src)
    }

    pub(crate) fn eat_path(&mut self) -> Option<tt::TokenTree> {
        self.eat_parse(ra_parser::parse_path)
    }

    pub(crate) fn expect_char(&mut self, char: char) -> Result<(), ParseError> {
        if self.at_char(char) {
            self.bump();
            Ok(())
        } else {
            Err(ParseError::Expected(format!("`{}`", char)))
        }
    }
}
