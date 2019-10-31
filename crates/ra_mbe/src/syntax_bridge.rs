//! FIXME: write short doc here

use ra_parser::{
    FragmentKind::{self, *},
    ParseError, TreeSink,
};
use ra_syntax::{
    ast, AstNode, AstToken, NodeOrToken, Parse, SmolStr, SyntaxKind, SyntaxKind::*, SyntaxNode,
    SyntaxTreeBuilder, TextRange, TextUnit, T,
};
use tt::buffer::{Cursor, TokenBuffer};

use crate::subtree_source::SubtreeTokenSource;
use crate::ExpandError;

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
    let tt = convert_tt(&mut token_map, node.text_range().start(), node)?;
    Some((tt, token_map))
}

/// Convert the syntax node to a `TokenTree` (what macro
/// will consume).
pub fn syntax_node_to_token_tree(node: &SyntaxNode) -> Option<(tt::Subtree, TokenMap)> {
    let mut token_map = TokenMap::default();
    let tt = convert_tt(&mut token_map, node.text_range().start(), node)?;
    Some((tt, token_map))
}

// The following items are what `rustc` macro can be parsed into :
// link: https://github.com/rust-lang/rust/blob/9ebf47851a357faa4cd97f4b1dc7835f6376e639/src/libsyntax/ext/expand.rs#L141
// * Expr(P<ast::Expr>)                     -> token_tree_to_expr
// * Pat(P<ast::Pat>)                       -> token_tree_to_pat
// * Ty(P<ast::Ty>)                         -> token_tree_to_ty
// * Stmts(SmallVec<[ast::Stmt; 1]>)        -> token_tree_to_stmts
// * Items(SmallVec<[P<ast::Item>; 1]>)     -> token_tree_to_items
//
// * TraitItems(SmallVec<[ast::TraitItem; 1]>)
// * ImplItems(SmallVec<[ast::ImplItem; 1]>)
// * ForeignItems(SmallVec<[ast::ForeignItem; 1]>

fn fragment_to_syntax_node(
    tt: &tt::Subtree,
    fragment_kind: FragmentKind,
) -> Result<Parse<SyntaxNode>, ExpandError> {
    let tmp;
    let tokens = match tt {
        tt::Subtree { delimiter: tt::Delimiter::None, token_trees } => token_trees.as_slice(),
        _ => {
            tmp = [tt.clone().into()];
            &tmp[..]
        }
    };
    let buffer = TokenBuffer::new(&tokens);
    let mut token_source = SubtreeTokenSource::new(&buffer);
    let mut tree_sink = TtTreeSink::new(buffer.begin());
    ra_parser::parse_fragment(&mut token_source, &mut tree_sink, fragment_kind);
    if tree_sink.roots.len() != 1 {
        return Err(ExpandError::ConversionError);
    }
    //FIXME: would be cool to report errors
    let parse = tree_sink.inner.finish();
    Ok(parse)
}

/// Parses the token tree (result of macro expansion) to an expression
pub fn token_tree_to_expr(tt: &tt::Subtree) -> Result<Parse<ast::Expr>, ExpandError> {
    let parse = fragment_to_syntax_node(tt, Expr)?;
    parse.cast().ok_or_else(|| crate::ExpandError::ConversionError)
}

/// Parses the token tree (result of macro expansion) to a Pattern
pub fn token_tree_to_pat(tt: &tt::Subtree) -> Result<Parse<ast::Pat>, ExpandError> {
    let parse = fragment_to_syntax_node(tt, Pattern)?;
    parse.cast().ok_or_else(|| crate::ExpandError::ConversionError)
}

/// Parses the token tree (result of macro expansion) to a Type
pub fn token_tree_to_ty(tt: &tt::Subtree) -> Result<Parse<ast::TypeRef>, ExpandError> {
    let parse = fragment_to_syntax_node(tt, Type)?;
    parse.cast().ok_or_else(|| crate::ExpandError::ConversionError)
}

/// Parses the token tree (result of macro expansion) as a sequence of stmts
pub fn token_tree_to_macro_stmts(tt: &tt::Subtree) -> Result<Parse<ast::MacroStmts>, ExpandError> {
    let parse = fragment_to_syntax_node(tt, Statements)?;
    parse.cast().ok_or_else(|| crate::ExpandError::ConversionError)
}

/// Parses the token tree (result of macro expansion) as a sequence of items
pub fn token_tree_to_items(tt: &tt::Subtree) -> Result<Parse<ast::MacroItems>, ExpandError> {
    let parse = fragment_to_syntax_node(tt, Items)?;
    parse.cast().ok_or_else(|| crate::ExpandError::ConversionError)
}

impl TokenMap {
    pub fn relative_range_of(&self, tt: tt::TokenId) -> Option<TextRange> {
        let idx = tt.0 as usize;
        self.tokens.get(idx).copied()
    }

    fn alloc(&mut self, relative_range: TextRange) -> tt::TokenId {
        let id = self.tokens.len();
        self.tokens.push(relative_range);
        tt::TokenId(id as u32)
    }
}

/// Returns the textual content of a doc comment block as a quoted string
/// That is, strips leading `///` (or `/**`, etc)
/// and strips the ending `*/`
/// And then quote the string, which is needed to convert to `tt::Literal`
fn doc_comment_text(comment: &ast::Comment) -> SmolStr {
    let prefix_len = comment.prefix().len();
    let mut text = &comment.text()[prefix_len..];

    // Remove ending "*/"
    if comment.kind().shape == ast::CommentShape::Block {
        text = &text[0..text.len() - 2];
    }

    // Quote the string
    // Note that `tt::Literal` expect an escaped string
    let text = format!("{:?}", text.escape_default().to_string());
    text.into()
}

fn convert_doc_comment(token: &ra_syntax::SyntaxToken) -> Option<Vec<tt::TokenTree>> {
    let comment = ast::Comment::cast(token.clone())?;
    let doc = comment.kind().doc?;

    // Make `doc="\" Comments\""
    let mut meta_tkns = Vec::new();
    meta_tkns.push(mk_ident("doc"));
    meta_tkns.push(mk_punct('='));
    meta_tkns.push(mk_doc_literal(&comment));

    // Make `#![]`
    let mut token_trees = Vec::new();
    token_trees.push(mk_punct('#'));
    if let ast::CommentPlacement::Inner = doc {
        token_trees.push(mk_punct('!'));
    }
    token_trees.push(tt::TokenTree::from(tt::Subtree {
        delimiter: tt::Delimiter::Bracket,
        token_trees: meta_tkns,
    }));

    return Some(token_trees);

    // Helper functions
    fn mk_ident(s: &str) -> tt::TokenTree {
        tt::TokenTree::from(tt::Leaf::from(tt::Ident {
            text: s.into(),
            id: tt::TokenId::unspecified(),
        }))
    }

    fn mk_punct(c: char) -> tt::TokenTree {
        tt::TokenTree::from(tt::Leaf::from(tt::Punct { char: c, spacing: tt::Spacing::Alone }))
    }

    fn mk_doc_literal(comment: &ast::Comment) -> tt::TokenTree {
        let lit = tt::Literal { text: doc_comment_text(comment) };

        tt::TokenTree::from(tt::Leaf::from(lit))
    }
}

fn convert_tt(
    token_map: &mut TokenMap,
    global_offset: TextUnit,
    tt: &SyntaxNode,
) -> Option<tt::Subtree> {
    // This tree is empty
    if tt.first_child_or_token().is_none() {
        return Some(tt::Subtree { token_trees: vec![], delimiter: tt::Delimiter::None });
    }

    let first_child = tt.first_child_or_token()?;
    let last_child = tt.last_child_or_token()?;
    let (delimiter, skip_first) = match (first_child.kind(), last_child.kind()) {
        (T!['('], T![')']) => (tt::Delimiter::Parenthesis, true),
        (T!['{'], T!['}']) => (tt::Delimiter::Brace, true),
        (T!['['], T![']']) => (tt::Delimiter::Bracket, true),
        _ => (tt::Delimiter::None, false),
    };

    let mut token_trees = Vec::new();
    let mut child_iter = tt.children_with_tokens().skip(skip_first as usize).peekable();

    while let Some(child) = child_iter.next() {
        if skip_first && (child == first_child || child == last_child) {
            continue;
        }

        match child {
            NodeOrToken::Token(token) => {
                if let Some(doc_tokens) = convert_doc_comment(&token) {
                    token_trees.extend(doc_tokens);
                } else if token.kind().is_trivia() {
                    continue;
                } else if token.kind().is_punct() {
                    assert!(token.text().len() == 1, "Input ast::token punct must be single char.");
                    let char = token.text().chars().next().unwrap();

                    let spacing = match child_iter.peek() {
                        Some(NodeOrToken::Token(token)) => {
                            if token.kind().is_punct() {
                                tt::Spacing::Joint
                            } else {
                                tt::Spacing::Alone
                            }
                        }
                        _ => tt::Spacing::Alone,
                    };

                    token_trees.push(tt::Leaf::from(tt::Punct { char, spacing }).into());
                } else {
                    let child: tt::TokenTree =
                        if token.kind() == T![true] || token.kind() == T![false] {
                            tt::Leaf::from(tt::Literal { text: token.text().clone() }).into()
                        } else if token.kind().is_keyword()
                            || token.kind() == IDENT
                            || token.kind() == LIFETIME
                        {
                            let relative_range = token.text_range() - global_offset;
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
            NodeOrToken::Node(node) => {
                let child = convert_tt(token_map, global_offset, &node)?.into();
                token_trees.push(child);
            }
        };
    }

    let res = tt::Subtree { delimiter, token_trees };
    Some(res)
}

struct TtTreeSink<'a> {
    buf: String,
    cursor: Cursor<'a>,
    text_pos: TextUnit,
    inner: SyntaxTreeBuilder,

    // Number of roots
    // Use for detect ill-form tree which is not single root
    roots: smallvec::SmallVec<[usize; 1]>,
}

impl<'a> TtTreeSink<'a> {
    fn new(cursor: Cursor<'a>) -> Self {
        TtTreeSink {
            buf: String::new(),
            cursor,
            text_pos: 0.into(),
            inner: SyntaxTreeBuilder::default(),
            roots: smallvec::SmallVec::new(),
        }
    }
}

fn delim_to_str(d: tt::Delimiter, closing: bool) -> SmolStr {
    let texts = match d {
        tt::Delimiter::Parenthesis => "()",
        tt::Delimiter::Brace => "{}",
        tt::Delimiter::Bracket => "[]",
        tt::Delimiter::None => "",
    };

    let idx = closing as usize;
    let text = if !texts.is_empty() { &texts[idx..texts.len() - (1 - idx)] } else { "" };
    text.into()
}

impl<'a> TreeSink for TtTreeSink<'a> {
    fn token(&mut self, kind: SyntaxKind, n_tokens: u8) {
        if kind == L_DOLLAR || kind == R_DOLLAR {
            self.cursor = self.cursor.bump_subtree();
            return;
        }

        for _ in 0..n_tokens {
            if self.cursor.eof() {
                break;
            }

            match self.cursor.token_tree() {
                Some(tt::TokenTree::Leaf(leaf)) => {
                    self.cursor = self.cursor.bump();
                    self.buf += &format!("{}", leaf);
                }
                Some(tt::TokenTree::Subtree(subtree)) => {
                    self.cursor = self.cursor.subtree().unwrap();
                    self.buf += &delim_to_str(subtree.delimiter, false);
                }
                None => {
                    if let Some(parent) = self.cursor.end() {
                        self.cursor = self.cursor.bump();
                        self.buf += &delim_to_str(parent.delimiter, true);
                    }
                }
            };
        }

        self.text_pos += TextUnit::of_str(&self.buf);
        let text = SmolStr::new(self.buf.as_str());
        self.buf.clear();
        self.inner.token(kind, text);

        // Add whitespace between adjoint puncts
        let next = self.cursor.bump();
        if let (
            Some(tt::TokenTree::Leaf(tt::Leaf::Punct(curr))),
            Some(tt::TokenTree::Leaf(tt::Leaf::Punct(_))),
        ) = (self.cursor.token_tree(), next.token_tree())
        {
            if curr.spacing == tt::Spacing::Alone {
                self.inner.token(WHITESPACE, " ".into());
            }
        }
    }

    fn start_node(&mut self, kind: SyntaxKind) {
        self.inner.start_node(kind);

        match self.roots.last_mut() {
            None | Some(0) => self.roots.push(1),
            Some(ref mut n) => **n += 1,
        };
    }

    fn finish_node(&mut self) {
        self.inner.finish_node();
        *self.roots.last_mut().unwrap() -= 1;
    }

    fn error(&mut self, error: ParseError) {
        self.inner.error(error, self.text_pos)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::{create_rules, expand};
    use ra_parser::TokenSource;

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
        let expansion = expand(&rules, "literals!(foo);");
        let tts = &[expansion.into()];
        let buffer = tt::buffer::TokenBuffer::new(tts);
        let mut tt_src = SubtreeTokenSource::new(&buffer);
        let mut tokens = vec![];
        while tt_src.current().kind != EOF {
            tokens.push((tt_src.current().kind, tt_src.text()));
            tt_src.bump();
        }

        // [${]
        // [let] [a] [=] ['c'] [;]
        assert_eq!(tokens[2 + 3].1, "'c'");
        assert_eq!(tokens[2 + 3].0, CHAR);
        // [let] [c] [=] [1000] [;]
        assert_eq!(tokens[2 + 5 + 3].1, "1000");
        assert_eq!(tokens[2 + 5 + 3].0, INT_NUMBER);
        // [let] [f] [=] [12E+99_f64] [;]
        assert_eq!(tokens[2 + 10 + 3].1, "12E+99_f64");
        assert_eq!(tokens[2 + 10 + 3].0, FLOAT_NUMBER);

        // [let] [s] [=] ["rust1"] [;]
        assert_eq!(tokens[2 + 15 + 3].1, "\"rust1\"");
        assert_eq!(tokens[2 + 15 + 3].0, STRING);
    }

    #[test]
    fn stmts_token_trees_to_expr_is_err() {
        let rules = create_rules(
            r#"
            macro_rules! stmts {
                () => {
                    let a = 0;
                    let b = 0;
                    let c = 0;
                    let d = 0;
                }
            }
            "#,
        );
        let expansion = expand(&rules, "stmts!();");
        assert!(token_tree_to_expr(&expansion).is_err());
    }
}
