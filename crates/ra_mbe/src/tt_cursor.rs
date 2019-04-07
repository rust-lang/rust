use crate::ParseError;
use crate::subtree_source::SubtreeTokenSource;

use ra_parser::{TokenSource, TreeSink};

use ra_syntax::{
    SyntaxKind
};

struct SubtreeTokenSink {
    token_pos: usize,
}

impl TreeSink for SubtreeTokenSink {
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
        src: &mut SubtreeTokenSource,
    ) -> Option<tt::TokenTree> {
        let (adv, res) = src.bump_n(parsed_token, self.pos);
        self.pos += adv;

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
        let mut src = SubtreeTokenSource::new(self.subtree);
        src.advance(self.pos, true);
        let mut sink = SubtreeTokenSink { token_pos: 0 };

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
