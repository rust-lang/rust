use crate::subtree_source::SubtreeTokenSource;

use ra_parser::{TokenSource, TreeSink};
use ra_syntax::{SyntaxKind};

struct OffsetTokenSink {
    token_pos: usize,
}

impl TreeSink for OffsetTokenSink {
    fn token(&mut self, _kind: SyntaxKind, n_tokens: u8) {
        self.token_pos += n_tokens as usize;
    }
    fn start_node(&mut self, _kind: SyntaxKind) {}
    fn finish_node(&mut self) {}
    fn error(&mut self, _error: ra_parser::ParseError) {}
}

pub(crate) struct Parser<'a> {
    subtree: &'a tt::Subtree,
    pos: &'a mut usize,
}

impl<'a> Parser<'a> {
    pub fn new(pos: &'a mut usize, subtree: &'a tt::Subtree) -> Parser<'a> {
        Parser { pos, subtree }
    }

    pub fn parse_path(self) -> Option<tt::TokenTree> {
        self.parse(ra_parser::parse_path)
    }

    fn parse<F>(self, f: F) -> Option<tt::TokenTree>
    where
        F: FnOnce(&dyn TokenSource, &mut dyn TreeSink),
    {
        let mut src = SubtreeTokenSource::new(self.subtree);
        src.advance(*self.pos, true);
        let mut sink = OffsetTokenSink { token_pos: 0 };

        f(&src, &mut sink);

        self.finish(sink.token_pos, &mut src)
    }

    fn finish(self, parsed_token: usize, src: &mut SubtreeTokenSource) -> Option<tt::TokenTree> {
        let res = src.bump_n(parsed_token, self.pos);
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
}
