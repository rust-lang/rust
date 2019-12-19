//! FIXME: write short doc here

use ra_parser::{FragmentKind, ParseError, TreeSink};
use ra_syntax::{
    ast, AstToken, NodeOrToken, Parse, SmolStr, SyntaxKind, SyntaxKind::*, SyntaxNode,
    SyntaxTreeBuilder, TextRange, TextUnit, T,
};
use rustc_hash::FxHashMap;
use std::iter::successors;
use tt::buffer::{Cursor, TokenBuffer};

use crate::subtree_source::SubtreeTokenSource;
use crate::ExpandError;

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum TokenTextRange {
    Token(TextRange),
    Delimiter(TextRange, TextRange),
}

impl TokenTextRange {
    pub fn by_kind(self, kind: SyntaxKind) -> Option<TextRange> {
        match self {
            TokenTextRange::Token(it) => Some(it),
            TokenTextRange::Delimiter(open, close) => match kind {
                T!['{'] | T!['('] | T!['['] => Some(open),
                T!['}'] | T![')'] | T![']'] => Some(close),
                _ => None,
            },
        }
    }
}

/// Maps `tt::TokenId` to the relative range of the original token.
#[derive(Debug, PartialEq, Eq, Default)]
pub struct TokenMap {
    /// Maps `tt::TokenId` to the *relative* source range.
    entries: Vec<(tt::TokenId, TokenTextRange)>,
}

/// Convert the syntax tree (what user has written) to a `TokenTree` (what macro
/// will consume).
pub fn ast_to_token_tree(ast: &impl ast::AstNode) -> Option<(tt::Subtree, TokenMap)> {
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
        tt::Subtree { delimiter: None, token_trees } => token_trees.as_slice(),
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
        let &(token_id, _) = self.entries.iter().find(|(_, range)| match range {
            TokenTextRange::Token(it) => *it == relative_range,
            TokenTextRange::Delimiter(open, close) => {
                *open == relative_range || *close == relative_range
            }
        })?;
        Some(token_id)
    }

    pub fn range_by_token(&self, token_id: tt::TokenId) -> Option<TokenTextRange> {
        let &(_, range) = self.entries.iter().find(|(tid, _)| *tid == token_id)?;
        Some(range)
    }

    fn insert(&mut self, token_id: tt::TokenId, relative_range: TextRange) {
        self.entries.push((token_id, TokenTextRange::Token(relative_range)));
    }

    fn insert_delim(
        &mut self,
        token_id: tt::TokenId,
        open_relative_range: TextRange,
        close_relative_range: TextRange,
    ) {
        self.entries
            .push((token_id, TokenTextRange::Delimiter(open_relative_range, close_relative_range)));
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
        delimiter: Some(tt::Delimiter {
            kind: tt::DelimiterKind::Bracket,
            id: tt::TokenId::unspecified(),
        }),
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
        tt::TokenTree::from(tt::Leaf::from(tt::Punct {
            char: c,
            spacing: tt::Spacing::Alone,
            id: tt::TokenId::unspecified(),
        }))
    }

    fn mk_doc_literal(comment: &ast::Comment) -> tt::TokenTree {
        let lit = tt::Literal { text: doc_comment_text(comment), id: tt::TokenId::unspecified() };

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
            return Some(tt::Subtree { token_trees: vec![], delimiter: None });
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
            return Some(tt::Subtree { token_trees: vec![], delimiter: None });
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

        let (delimiter_kind, skip_first) = match (first_child.kind(), last_child.kind()) {
            (T!['('], T![')']) => (Some(tt::DelimiterKind::Parenthesis), true),
            (T!['{'], T!['}']) => (Some(tt::DelimiterKind::Brace), true),
            (T!['['], T![']']) => (Some(tt::DelimiterKind::Bracket), true),
            _ => (None, false),
        };
        let delimiter = delimiter_kind.map(|kind| tt::Delimiter {
            kind,
            id: self.alloc_delim(first_child.text_range(), last_child.text_range()),
        });

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
                        // we need to pull apart joined punctuation tokens
                        let last_spacing = match child_iter.peek() {
                            Some(NodeOrToken::Token(token)) => {
                                if token.kind().is_punct() {
                                    tt::Spacing::Joint
                                } else {
                                    tt::Spacing::Alone
                                }
                            }
                            _ => tt::Spacing::Alone,
                        };
                        let spacing_iter = std::iter::repeat(tt::Spacing::Joint)
                            .take(token.text().len() - 1)
                            .chain(std::iter::once(last_spacing));
                        for (char, spacing) in token.text().chars().zip(spacing_iter) {
                            token_trees.push(
                                tt::Leaf::from(tt::Punct {
                                    char,
                                    spacing,
                                    id: self.alloc(token.text_range()),
                                })
                                .into(),
                            );
                        }
                    } else {
                        macro_rules! make_leaf {
                            ($i:ident) => {
                                tt::$i {
                                    id: self.alloc(token.text_range()),
                                    text: token.text().clone(),
                                }
                                .into()
                            };
                        }

                        let child: tt::Leaf = match token.kind() {
                            T![true] | T![false] => make_leaf!(Literal),
                            IDENT | LIFETIME => make_leaf!(Ident),
                            k if k.is_keyword() => make_leaf!(Ident),
                            k if k.is_literal() => make_leaf!(Literal),
                            _ => return None,
                        };
                        token_trees.push(child.into());
                    }
                }
                NodeOrToken::Node(node) => {
                    let child_subtree = self.go(&node)?;
                    if child_subtree.delimiter.is_none() && node.kind() != SyntaxKind::TOKEN_TREE {
                        token_trees.extend(child_subtree.token_trees);
                    } else {
                        token_trees.push(child_subtree.into());
                    }
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

    fn alloc_delim(
        &mut self,
        open_abs_range: TextRange,
        close_abs_range: TextRange,
    ) -> tt::TokenId {
        let open_relative_range = open_abs_range - self.global_offset;
        let close_relative_range = close_abs_range - self.global_offset;
        let token_id = tt::TokenId(self.next_id);
        self.next_id += 1;

        self.map.insert_delim(token_id, open_relative_range, close_relative_range);
        token_id
    }
}

struct TtTreeSink<'a> {
    buf: String,
    cursor: Cursor<'a>,
    open_delims: FxHashMap<tt::TokenId, TextUnit>,
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
            open_delims: FxHashMap::default(),
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

fn delim_to_str(d: Option<tt::DelimiterKind>, closing: bool) -> SmolStr {
    let texts = match d {
        Some(tt::DelimiterKind::Parenthesis) => "()",
        Some(tt::DelimiterKind::Brace) => "{}",
        Some(tt::DelimiterKind::Bracket) => "[]",
        None => return "".into(),
    };

    let idx = closing as usize;
    let text = &texts[idx..texts.len() - (1 - idx)];
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

            let text: SmolStr = match self.cursor.token_tree() {
                Some(tt::TokenTree::Leaf(leaf)) => {
                    // Mark the range if needed
                    let id = match leaf {
                        tt::Leaf::Ident(ident) => ident.id,
                        tt::Leaf::Punct(punct) => punct.id,
                        tt::Leaf::Literal(lit) => lit.id,
                    };
                    let text = SmolStr::new(format!("{}", leaf));
                    let range = TextRange::offset_len(self.text_pos, TextUnit::of_str(&text));
                    self.token_map.insert(id, range);
                    self.cursor = self.cursor.bump();
                    text
                }
                Some(tt::TokenTree::Subtree(subtree)) => {
                    self.cursor = self.cursor.subtree().unwrap();
                    if let Some(id) = subtree.delimiter.map(|it| it.id) {
                        self.open_delims.insert(id, self.text_pos);
                    }
                    delim_to_str(subtree.delimiter_kind(), false)
                }
                None => {
                    if let Some(parent) = self.cursor.end() {
                        self.cursor = self.cursor.bump();
                        if let Some(id) = parent.delimiter.map(|it| it.id) {
                            if let Some(open_delim) = self.open_delims.get(&id) {
                                let open_range =
                                    TextRange::offset_len(*open_delim, TextUnit::from_usize(1));
                                let close_range =
                                    TextRange::offset_len(self.text_pos, TextUnit::from_usize(1));
                                self.token_map.insert_delim(id, open_range, close_range);
                            }
                        }
                        delim_to_str(parent.delimiter_kind(), true)
                    } else {
                        continue;
                    }
                }
            };
            self.buf += &text;
            self.text_pos += TextUnit::of_str(&text);
        }

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
    use crate::tests::parse_macro;
    use ra_parser::TokenSource;
    use ra_syntax::{
        algo::{insert_children, InsertPosition},
        ast::AstNode,
    };

    #[test]
    fn convert_tt_token_source() {
        let expansion = parse_macro(
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
        )
        .expand_tt("literals!(foo);");
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
        let expansion = parse_macro(
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
        )
        .expand_tt("stmts!();");
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

        assert_eq!(tt.delimiter_kind(), Some(tt::DelimiterKind::Brace));
    }

    #[test]
    fn test_token_tree_multi_char_punct() {
        let source_file = ast::SourceFile::parse("struct Foo { a: x::Y }").ok().unwrap();
        let struct_def = source_file.syntax().descendants().find_map(ast::StructDef::cast).unwrap();
        let tt = ast_to_token_tree(&struct_def).unwrap().0;
        token_tree_to_syntax_node(&tt, FragmentKind::Item).unwrap();
    }
}
