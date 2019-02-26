use ra_parser::{TokenSource, TreeSink, ParseError};
use ra_syntax::{
    AstNode, SyntaxNode, TextRange, SyntaxKind, SmolStr, SyntaxTreeBuilder, TreeArc,
    ast, SyntaxKind::*, TextUnit
};

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
    let token_source = TtTokenSource::new(tt);
    let mut tree_sink = TtTreeSink::new(&token_source.tokens);
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
    let first_child = tt.first_child()?;
    let last_child = tt.last_child()?;
    let delimiter = match (first_child.kind(), last_child.kind()) {
        (L_PAREN, R_PAREN) => tt::Delimiter::Parenthesis,
        (L_CURLY, R_CURLY) => tt::Delimiter::Brace,
        (L_BRACK, R_BRACK) => tt::Delimiter::Bracket,
        _ => return None,
    };
    let mut token_trees = Vec::new();
    for child in tt.children().skip(1) {
        if child == first_child || child == last_child || child.kind().is_trivia() {
            continue;
        }
        if child.kind().is_punct() {
            let mut prev = None;
            for char in child.leaf_text().unwrap().chars() {
                if let Some(char) = prev {
                    token_trees.push(
                        tt::Leaf::from(tt::Punct { char, spacing: tt::Spacing::Joint }).into(),
                    );
                }
                prev = Some(char)
            }
            if let Some(char) = prev {
                token_trees
                    .push(tt::Leaf::from(tt::Punct { char, spacing: tt::Spacing::Alone }).into());
            }
        } else {
            let child: tt::TokenTree = if child.kind() == TOKEN_TREE {
                convert_tt(token_map, global_offset, child)?.into()
            } else if child.kind().is_keyword() || child.kind() == IDENT {
                let relative_range = child.range() - global_offset;
                let id = token_map.alloc(relative_range);
                let text = child.leaf_text().unwrap().clone();
                tt::Leaf::from(tt::Ident { text, id }).into()
            } else if child.kind().is_literal() {
                tt::Leaf::from(tt::Literal { text: child.leaf_text().unwrap().clone() }).into()
            } else {
                return None;
            };
            token_trees.push(child)
        }
    }

    let res = tt::Subtree { delimiter, token_trees };
    Some(res)
}

struct TtTokenSource {
    tokens: Vec<TtToken>,
}

struct TtToken {
    kind: SyntaxKind,
    is_joint_to_next: bool,
    text: SmolStr,
}

impl TtTokenSource {
    fn new(tt: &tt::Subtree) -> TtTokenSource {
        let mut res = TtTokenSource { tokens: Vec::new() };
        res.convert_subtree(tt);
        res
    }
    fn convert_subtree(&mut self, sub: &tt::Subtree) {
        self.push_delim(sub.delimiter, false);
        sub.token_trees.iter().for_each(|tt| self.convert_tt(tt));
        self.push_delim(sub.delimiter, true)
    }
    fn convert_tt(&mut self, tt: &tt::TokenTree) {
        match tt {
            tt::TokenTree::Leaf(leaf) => self.convert_leaf(leaf),
            tt::TokenTree::Subtree(sub) => self.convert_subtree(sub),
        }
    }
    fn convert_leaf(&mut self, leaf: &tt::Leaf) {
        let tok = match leaf {
            tt::Leaf::Literal(l) => TtToken {
                kind: SyntaxKind::INT_NUMBER, // FIXME
                is_joint_to_next: false,
                text: l.text.clone(),
            },
            tt::Leaf::Punct(p) => {
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
                TtToken { kind, is_joint_to_next: p.spacing == tt::Spacing::Joint, text }
            }
            tt::Leaf::Ident(ident) => {
                let kind = SyntaxKind::from_keyword(ident.text.as_str()).unwrap_or(IDENT);
                TtToken { kind, is_joint_to_next: false, text: ident.text.clone() }
            }
        };
        self.tokens.push(tok)
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
        let tok = TtToken { kind, is_joint_to_next: false, text: SmolStr::new(text) };
        self.tokens.push(tok)
    }
}

impl TokenSource for TtTokenSource {
    fn token_kind(&self, pos: usize) -> SyntaxKind {
        if let Some(tok) = self.tokens.get(pos) {
            tok.kind
        } else {
            SyntaxKind::EOF
        }
    }
    fn is_token_joint_to_next(&self, pos: usize) -> bool {
        self.tokens[pos].is_joint_to_next
    }
    fn is_keyword(&self, pos: usize, kw: &str) -> bool {
        self.tokens[pos].text == *kw
    }
}

#[derive(Default)]
struct TtTreeSink<'a> {
    buf: String,
    tokens: &'a [TtToken],
    text_pos: TextUnit,
    token_pos: usize,
    inner: SyntaxTreeBuilder,
}

impl<'a> TtTreeSink<'a> {
    fn new(tokens: &'a [TtToken]) -> TtTreeSink {
        TtTreeSink {
            buf: String::new(),
            tokens,
            text_pos: 0.into(),
            token_pos: 0,
            inner: SyntaxTreeBuilder::default(),
        }
    }
}

impl<'a> TreeSink for TtTreeSink<'a> {
    fn leaf(&mut self, kind: SyntaxKind, n_tokens: u8) {
        for _ in 0..n_tokens {
            self.buf += self.tokens[self.token_pos].text.as_str();
            self.token_pos += 1;
        }
        self.text_pos += TextUnit::of_str(&self.buf);
        let text = SmolStr::new(self.buf.as_str());
        self.buf.clear();
        self.inner.leaf(kind, text)
    }

    fn start_branch(&mut self, kind: SyntaxKind) {
        self.inner.start_branch(kind);
    }

    fn finish_branch(&mut self) {
        self.inner.finish_branch();
    }

    fn error(&mut self, error: ParseError) {
        self.inner.error(error, self.text_pos)
    }
}
