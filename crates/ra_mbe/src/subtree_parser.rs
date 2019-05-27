use crate::subtree_source::SubtreeTokenSource;

use ra_parser::{TokenSource, TreeSink};
use ra_syntax::{SyntaxKind};
use tt::buffer::{TokenBuffer, Cursor};

struct OffsetTokenSink<'a> {
    cursor: Cursor<'a>,
    error: bool,
}

impl<'a> OffsetTokenSink<'a> {
    pub fn collect(&self, begin: Cursor<'a>) -> Vec<tt::TokenTree> {
        if !self.cursor.is_root() {
            return vec![];
        }

        let mut curr = begin;
        let mut res = vec![];

        while self.cursor != curr {
            if let Some(token) = curr.token_tree() {
                res.push(token);
            }
            curr = curr.bump();
        }

        res
    }
}

impl<'a> TreeSink for OffsetTokenSink<'a> {
    fn token(&mut self, _kind: SyntaxKind, n_tokens: u8) {
        for _ in 0..n_tokens {
            self.cursor = self.cursor.bump_subtree();
        }
    }
    fn start_node(&mut self, _kind: SyntaxKind) {}
    fn finish_node(&mut self) {}
    fn error(&mut self, _error: ra_parser::ParseError) {
        self.error = true;
    }
}

pub(crate) struct Parser<'a> {
    subtree: &'a tt::Subtree,
    cur_pos: &'a mut usize,
}

impl<'a> Parser<'a> {
    pub fn new(cur_pos: &'a mut usize, subtree: &'a tt::Subtree) -> Parser<'a> {
        Parser { cur_pos, subtree }
    }

    pub fn parse_path(self) -> Option<tt::TokenTree> {
        self.parse(ra_parser::parse_path)
    }

    pub fn parse_expr(self) -> Option<tt::TokenTree> {
        self.parse(ra_parser::parse_expr)
    }

    pub fn parse_ty(self) -> Option<tt::TokenTree> {
        self.parse(ra_parser::parse_ty)
    }

    pub fn parse_pat(self) -> Option<tt::TokenTree> {
        self.parse(ra_parser::parse_pat)
    }

    pub fn parse_stmt(self) -> Option<tt::TokenTree> {
        self.parse(|src, sink| ra_parser::parse_stmt(src, sink, false))
    }

    pub fn parse_block(self) -> Option<tt::TokenTree> {
        self.parse(ra_parser::parse_block)
    }

    pub fn parse_meta(self) -> Option<tt::TokenTree> {
        self.parse(ra_parser::parse_meta)
    }

    pub fn parse_item(self) -> Option<tt::TokenTree> {
        self.parse(ra_parser::parse_item)
    }

    pub fn parse_vis(self) -> Option<tt::TokenTree> {
        self.parse(ra_parser::parse_vis)
    }

    fn parse<F>(self, f: F) -> Option<tt::TokenTree>
    where
        F: FnOnce(&mut dyn TokenSource, &mut dyn TreeSink),
    {
        let buffer = TokenBuffer::new(&self.subtree.token_trees[*self.cur_pos..]);
        let mut src = SubtreeTokenSource::new(&buffer);
        let mut sink = OffsetTokenSink { cursor: buffer.begin(), error: false };

        f(&mut src, &mut sink);

        let r = self.finish(buffer.begin(), &mut sink);
        if sink.error {
            return None;
        }
        r
    }

    fn finish(self, begin: Cursor, sink: &mut OffsetTokenSink) -> Option<tt::TokenTree> {
        let res = sink.collect(begin);
        *self.cur_pos += res.len();

        match res.len() {
            0 => None,
            1 => Some(res[0].clone()),
            _ => Some(tt::TokenTree::Subtree(tt::Subtree {
                delimiter: tt::Delimiter::None,
                token_trees: res,
            })),
        }
    }
}
