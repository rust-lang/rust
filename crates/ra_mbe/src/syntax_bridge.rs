//! FIXME: write short doc here

use ra_parser::{FragmentKind, ParseError, TreeSink};
use ra_syntax::{
    ast, AstNode, AstToken, NodeOrToken, Parse, SmolStr, SyntaxKind, SyntaxKind::*, SyntaxNode,
    SyntaxTreeBuilder, TextRange, TextUnit, T,
};
use std::iter::successors;
use tt::buffer::{Cursor, TokenBuffer};

use crate::subtree_source::SubtreeTokenSource;
use crate::ExpandError;

/// Maps `tt::TokenId` to the relative range of the original token.
#[derive(Debug, PartialEq, Eq, Default)]
pub struct TokenMap {
    /// Maps `tt::TokenId` to the *relative* source range.
    entries: Vec<(tt::TokenId, TextRange)>,
}

/// Convert the syntax tree (what user has written) to a `TokenTree` (what macro
/// will consume).
pub fn ast_to_token_tree(ast: &ast::TokenTree) -> Option<(tt::Subtree, TokenMap)> {
    syntax_node_to_token_tree(ast.syntax())
}

/// Convert the syntax node to a `TokenTree` (what macro
/// will consume).
pub fn syntax_node_to_token_tree(node: &SyntaxNode) -> Option<(tt::Subtree, TokenMap)> {
    let global_offset = node.text_range().start();
    let mut c = Convertor { map: TokenMap::default(), global_offset, next_id: 0 };
    let subtree = c.go(node)?;
    Some((subtree, c.map))
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

pub fn token_tree_to_syntax_node(
    tt: &tt::Subtree,
    fragment_kind: FragmentKind,
) -> Result<(Parse<SyntaxNode>, TokenMap), ExpandError> {
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
    let (parse, range_map) = tree_sink.finish();
    Ok((parse, range_map))
}

impl TokenMap {
    pub fn token_by_range(&self, relative_range: TextRange) -> Option<tt::TokenId> {
        let &(token_id, _) = self.entries.iter().find(|(_, range)| *range == relative_range)?;
        Some(token_id)
    }

    pub fn range_by_token(&self, token_id: tt::TokenId) -> Option<TextRange> {
        let &(_, range) = self.entries.iter().find(|(tid, _)| *tid == token_id)?;
        Some(range)
    }

    fn insert(&mut self, token_id: tt::TokenId, relative_range: TextRange) {
        self.entries.push((token_id, relative_range));
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

struct Convertor {
    map: TokenMap,
    global_offset: TextUnit,
    next_id: u32,
}

impl Convertor {
    fn go(&mut self, tt: &SyntaxNode) -> Option<tt::Subtree> {
        // This tree is empty
        if tt.first_child_or_token().is_none() {
            return Some(tt::Subtree { token_trees: vec![], delimiter: tt::Delimiter::None });
        }

        let first_child = tt.first_child_or_token()?;
        let last_child = tt.last_child_or_token()?;

        // ignore trivial first_child and last_child
        let first_child = successors(Some(first_child), |it| {
            if it.kind().is_trivia() {
                it.next_sibling_or_token()
            } else {
                None
            }
        })
        .last()
        .unwrap();
        if first_child.kind().is_trivia() {
            return Some(tt::Subtree { token_trees: vec![], delimiter: tt::Delimiter::None });
        }

        let last_child = successors(Some(last_child), |it| {
            if it.kind().is_trivia() {
                it.prev_sibling_or_token()
            } else {
                None
            }
        })
        .last()
        .unwrap();

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
                        assert!(
                            token.text().len() == 1,
                            "Input ast::token punct must be single char."
                        );
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
                                let id = self.alloc(token.text_range());
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
                    let child = self.go(&node)?.into();
                    token_trees.push(child);
                }
            };
        }

        let res = tt::Subtree { delimiter, token_trees };
        Some(res)
    }

    fn alloc(&mut self, absolute_range: TextRange) -> tt::TokenId {
        let relative_range = absolute_range - self.global_offset;
        let token_id = tt::TokenId(self.next_id);
        self.next_id += 1;
        self.map.insert(token_id, relative_range);
        token_id
    }
}

struct TtTreeSink<'a> {
    buf: String,
    cursor: Cursor<'a>,
    text_pos: TextUnit,
    inner: SyntaxTreeBuilder,
    token_map: TokenMap,

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
            token_map: TokenMap::default(),
        }
    }

    fn finish(self) -> (Parse<SyntaxNode>, TokenMap) {
        (self.inner.finish(), self.token_map)
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
                    // Mark the range if needed
                    if let tt::Leaf::Ident(ident) = leaf {
                        if kind == IDENT {
                            let range =
                                TextRange::offset_len(self.text_pos, TextUnit::of_str(&ident.text));
                            self.token_map.insert(ident.id, range);
                        }
                    }

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
                self.text_pos += TextUnit::of_char(' ');
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
    use ra_syntax::algo::{insert_children, InsertPosition};

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
        assert!(token_tree_to_syntax_node(&expansion, FragmentKind::Expr).is_err());
    }

    #[test]
    fn test_token_tree_last_child_is_white_space() {
        let source_file = ast::SourceFile::parse("f!({} );").ok().unwrap();
        let macro_call = source_file.syntax().descendants().find_map(ast::MacroCall::cast).unwrap();
        let token_tree = macro_call.token_tree().unwrap();

        // Token Tree now is :
        // TokenTree
        // - T!['(']
        // - TokenTree
        //   - T!['{']
        //   - T!['}']
        // - WHITE_SPACE
        // - T![')']

        let rbrace =
            token_tree.syntax().descendants_with_tokens().find(|it| it.kind() == T!['}']).unwrap();
        let space = token_tree
            .syntax()
            .descendants_with_tokens()
            .find(|it| it.kind() == SyntaxKind::WHITESPACE)
            .unwrap();

        // reorder th white space, such that the white is inside the inner token-tree.
        let token_tree = insert_children(
            &rbrace.parent().unwrap(),
            InsertPosition::Last,
            &mut std::iter::once(space),
        );

        // Token Tree now is :
        // TokenTree
        // - T!['{']
        // - T!['}']
        // - WHITE_SPACE
        let token_tree = ast::TokenTree::cast(token_tree).unwrap();
        let tt = ast_to_token_tree(&token_tree).unwrap().0;

        assert_eq!(tt.delimiter, tt::Delimiter::Brace);
    }
}
