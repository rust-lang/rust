use ra_parser::{TreeSink, ParseError};
use ra_syntax::{
    AstNode, SyntaxNode, TextRange, SyntaxKind, SmolStr, SyntaxTreeBuilder, TreeArc, SyntaxElement,
    ast, SyntaxKind::*, TextUnit
};

use crate::subtree_source::{SubtreeTokenSource, SubtreeSourceQuerier};

/// Maps `tt::TokenId` to the relative range of the original token.
#[derive(Default)]
pub struct TokenMap {
    /// Maps `tt::TokenId` to the *relative* source range.
    tokens: Vec<TextRange>,
}

/// Convert the syntax tree (what user has written) to a `TokenTree` (what macro
/// will consume).
pub fn ast_to_token_tree(ast: &ast::TokenTree) -> Option<(tt::Subtree, TokenMap)> {
    let mut token_map = TokenMap::default();
    let node = ast.syntax();
    let tt = convert_tt(&mut token_map, node.range().start(), node)?;
    Some((tt, token_map))
}

/// Parses the token tree (result of macro expansion) as a sequence of items
pub fn token_tree_to_ast_item_list(tt: &tt::Subtree) -> TreeArc<ast::SourceFile> {
    let token_source = SubtreeTokenSource::new(tt);
    let mut tree_sink = TtTreeSink::new(token_source.querier());
    ra_parser::parse(&token_source, &mut tree_sink);
    let syntax = tree_sink.inner.finish();
    ast::SourceFile::cast(&syntax).unwrap().to_owned()
}

impl TokenMap {
    pub fn relative_range_of(&self, tt: tt::TokenId) -> Option<TextRange> {
        let idx = tt.0 as usize;
        self.tokens.get(idx).map(|&it| it)
    }

    fn alloc(&mut self, relative_range: TextRange) -> tt::TokenId {
        let id = self.tokens.len();
        self.tokens.push(relative_range);
        tt::TokenId(id as u32)
    }
}

fn convert_tt(
    token_map: &mut TokenMap,
    global_offset: TextUnit,
    tt: &SyntaxNode,
) -> Option<tt::Subtree> {
    let first_child = tt.first_child_or_token()?;
    let last_child = tt.last_child_or_token()?;
    let delimiter = match (first_child.kind(), last_child.kind()) {
        (L_PAREN, R_PAREN) => tt::Delimiter::Parenthesis,
        (L_CURLY, R_CURLY) => tt::Delimiter::Brace,
        (L_BRACK, R_BRACK) => tt::Delimiter::Bracket,
        _ => return None,
    };
    let mut token_trees = Vec::new();
    for child in tt.children_with_tokens().skip(1) {
        if child == first_child || child == last_child || child.kind().is_trivia() {
            continue;
        }
        match child {
            SyntaxElement::Token(token) => {
                if token.kind().is_punct() {
                    let mut prev = None;
                    for char in token.text().chars() {
                        if let Some(char) = prev {
                            token_trees.push(
                                tt::Leaf::from(tt::Punct { char, spacing: tt::Spacing::Joint })
                                    .into(),
                            );
                        }
                        prev = Some(char)
                    }
                    if let Some(char) = prev {
                        token_trees.push(
                            tt::Leaf::from(tt::Punct { char, spacing: tt::Spacing::Alone }).into(),
                        );
                    }
                } else {
                    let child = if token.kind().is_keyword() || token.kind() == IDENT {
                        let relative_range = token.range() - global_offset;
                        let id = token_map.alloc(relative_range);
                        let text = token.text().clone();
                        tt::Leaf::from(tt::Ident { text, id }).into()
                    } else if token.kind().is_literal() {
                        tt::Leaf::from(tt::Literal { text: token.text().clone() }).into()
                    } else {
                        return None;
                    };
                    token_trees.push(child);
                }
            }
            SyntaxElement::Node(node) => {
                let child = convert_tt(token_map, global_offset, node)?.into();
                token_trees.push(child);
            }
        };
    }

    let res = tt::Subtree { delimiter, token_trees };
    Some(res)
}

struct TtTreeSink<'a> {
    buf: String,
    src_querier: SubtreeSourceQuerier<'a>,
    text_pos: TextUnit,
    token_pos: usize,
    inner: SyntaxTreeBuilder,
}

impl<'a> TtTreeSink<'a> {
    fn new(src_querier: SubtreeSourceQuerier<'a>) -> TtTreeSink {
        TtTreeSink {
            buf: String::new(),
            src_querier,
            text_pos: 0.into(),
            token_pos: 0,
            inner: SyntaxTreeBuilder::default(),
        }
    }
}

impl<'a> TreeSink for TtTreeSink<'a> {
    fn token(&mut self, kind: SyntaxKind, n_tokens: u8) {
        for _ in 0..n_tokens {
            self.buf += self.src_querier.token(self.token_pos).1;
            self.token_pos += 1;
        }
        self.text_pos += TextUnit::of_str(&self.buf);
        let text = SmolStr::new(self.buf.as_str());
        self.buf.clear();
        self.inner.token(kind, text)
    }

    fn start_node(&mut self, kind: SyntaxKind) {
        self.inner.start_node(kind);
    }

    fn finish_node(&mut self) {
        self.inner.finish_node();
    }

    fn error(&mut self, error: ParseError) {
        self.inner.error(error, self.text_pos)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::{expand, create_rules};

    #[test]
    fn convert_tt_token_source() {
        let rules = create_rules(
            r#"
            macro_rules! literals {
                ($i:ident) => {
                    {
                        let a = 'c';
                        let c = 1000;
                        let f = 12E+99_f64;
                        let s = "rust1";
                    }
                }
            }
            "#,
        );
        let expansion = expand(&rules, "literals!(foo)");
        let tt_src = SubtreeTokenSource::new(&expansion);

        let query = tt_src.querier();

        // [{]
        // [let] [a] [=] ['c'] [;]
        assert_eq!(query.token(1 + 3).1, "'c'");
        assert_eq!(query.token(1 + 3).0, CHAR);
        // [let] [c] [=] [1000] [;]
        assert_eq!(query.token(1 + 5 + 3).1, "1000");
        assert_eq!(query.token(1 + 5 + 3).0, INT_NUMBER);
        // [let] [f] [=] [12E+99_f64] [;]
        assert_eq!(query.token(1 + 10 + 3).1, "12E+99_f64");
        assert_eq!(query.token(1 + 10 + 3).0, FLOAT_NUMBER);

        // [let] [s] [=] ["rust1"] [;]
        assert_eq!(query.token(1 + 15 + 3).1, "\"rust1\"");
        assert_eq!(query.token(1 + 15 + 3).0, STRING);
    }
}
