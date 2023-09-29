//! Conversions between [`SyntaxNode`] and [`tt::TokenTree`].

use stdx::non_empty_vec::NonEmptyVec;
use syntax::{
    ast::{self, make::tokens::doc_comment},
    AstToken, NodeOrToken, Parse, PreorderWithTokens, SmolStr, SyntaxElement, SyntaxKind,
    SyntaxKind::*,
    SyntaxNode, SyntaxToken, SyntaxTreeBuilder, TextRange, TextSize, WalkEvent, T,
};
use tt::{
    buffer::{Cursor, TokenBuffer},
    Span, SpanData,
};

use crate::{to_parser_input::to_parser_input, tt_iter::TtIter, TokenMap};

#[cfg(test)]
mod tests;

/// Convert the syntax node to a `TokenTree` (what macro
/// will consume).
/// `anchor` and `anchor_offset` are used to convert the node's spans
/// to relative spans, relative to the passed anchor.
/// `map` is used to resolve the converted spans accordingly.
/// TODO: Flesh out the doc comment more thoroughly
pub fn syntax_node_to_token_tree<SpanAnchor: Copy>(
    node: &SyntaxNode,
    anchor: SpanAnchor,
    anchor_offset: TextSize,
    map: &TokenMap<SpanData<SpanAnchor>>,
) -> tt::Subtree<SpanData<SpanAnchor>>
where
    SpanData<SpanAnchor>: Span,
{
    assert!(anchor_offset <= node.text_range().start());
    let mut c = Converter::new(node, anchor_offset, anchor, vec![], map);
    convert_tokens(&mut c)
}

pub fn syntax_node_to_token_tree_censored<SpanAnchor: Copy>(
    node: &SyntaxNode,
    anchor: SpanAnchor,
    anchor_offset: TextSize,
    map: &TokenMap<SpanData<SpanAnchor>>,
    censored: Vec<SyntaxNode>,
) -> tt::Subtree<SpanData<SpanAnchor>>
where
    SpanData<SpanAnchor>: Span,
{
    assert!(anchor_offset <= node.text_range().start());
    let mut c = Converter::new(node, anchor_offset, anchor, censored, map);
    convert_tokens(&mut c)
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
// * AssocItems(SmallVec<[ast::AssocItem; 1]>)
// * ForeignItems(SmallVec<[ast::ForeignItem; 1]>

pub fn token_tree_to_syntax_node<SpanAnchor: Copy>(
    tt: &tt::Subtree<SpanData<SpanAnchor>>,
    entry_point: parser::TopEntryPoint,
) -> (Parse<SyntaxNode>, TokenMap<SpanData<SpanAnchor>>)
where
    SpanData<SpanAnchor>: Span,
{
    let buffer = match tt {
        tt::Subtree {
            delimiter: tt::Delimiter { kind: tt::DelimiterKind::Invisible, .. },
            token_trees,
        } => TokenBuffer::from_tokens(token_trees.as_slice()),
        _ => TokenBuffer::from_subtree(tt),
    };
    let parser_input = to_parser_input(&buffer);
    let parser_output = entry_point.parse(&parser_input);
    let mut tree_sink = TtTreeSink::new(buffer.begin());
    for event in parser_output.iter() {
        match event {
            parser::Step::Token { kind, n_input_tokens: n_raw_tokens } => {
                tree_sink.token(kind, n_raw_tokens)
            }
            parser::Step::FloatSplit { ends_in_dot: has_pseudo_dot } => {
                tree_sink.float_split(has_pseudo_dot)
            }
            parser::Step::Enter { kind } => tree_sink.start_node(kind),
            parser::Step::Exit => tree_sink.finish_node(),
            parser::Step::Error { msg } => tree_sink.error(msg.to_string()),
        }
    }
    tree_sink.finish()
}

pub fn map_from_syntax_node<SpanAnchor>(
    node: &SyntaxNode,
    anchor: SpanAnchor,
    anchor_offset: TextSize,
) -> TokenMap<SpanData<SpanAnchor>>
where
    SpanAnchor: Copy,
    SpanData<SpanAnchor>: Span,
{
    let mut map = TokenMap::default();
    node.descendants_with_tokens().filter_map(NodeOrToken::into_token).for_each(|t| {
        map.insert(t.text_range(), SpanData { range: t.text_range() - anchor_offset, anchor });
    });
    map
}

/// Convert a string to a `TokenTree`
pub fn parse_to_token_tree<SpanAnchor: Copy>(
    text: &str,
    file_id: SpanAnchor,
) -> Option<tt::Subtree<SpanData<SpanAnchor>>>
where
    SpanData<SpanAnchor>: Span,
{
    let lexed = parser::LexedStr::new(text);
    if lexed.errors().next().is_some() {
        return None;
    }
    let mut conv = RawConverter { lexed, pos: 0, _offset: TextSize::default(), file_id };
    Some(convert_tokens(&mut conv))
}

/// Split token tree with separate expr: $($e:expr)SEP*
pub fn parse_exprs_with_sep<S: Span>(tt: &tt::Subtree<S>, sep: char) -> Vec<tt::Subtree<S>> {
    if tt.token_trees.is_empty() {
        return Vec::new();
    }

    let mut iter = TtIter::new(tt);
    let mut res = Vec::new();

    while iter.peek_n(0).is_some() {
        let expanded = iter.expect_fragment(parser::PrefixEntryPoint::Expr);

        res.push(match expanded.value {
            None => break,
            Some(tt @ tt::TokenTree::Leaf(_)) => {
                tt::Subtree { delimiter: tt::Delimiter::unspecified(), token_trees: vec![tt] }
            }
            Some(tt::TokenTree::Subtree(tt)) => tt,
        });

        let mut fork = iter.clone();
        if fork.expect_char(sep).is_err() {
            break;
        }
        iter = fork;
    }

    if iter.peek_n(0).is_some() {
        res.push(tt::Subtree {
            delimiter: tt::Delimiter::unspecified(),
            token_trees: iter.cloned().collect(),
        });
    }

    res
}

fn convert_tokens<SpanAnchor, C: TokenConverter<SpanAnchor>>(
    conv: &mut C,
) -> tt::Subtree<SpanData<SpanAnchor>>
where
    SpanData<SpanAnchor>: Span,
    SpanAnchor: Copy,
{
    let entry = tt::Subtree { delimiter: tt::Delimiter::unspecified(), token_trees: vec![] };
    let mut stack = NonEmptyVec::new(entry);
    let anchor = conv.anchor();

    loop {
        let subtree = stack.last_mut();
        let result = &mut subtree.token_trees;
        let Some((token, rel_range, abs_range)) = conv.bump() else { break };

        let kind = token.kind(conv);
        if kind == COMMENT {
            if let Some(tokens) = conv.convert_doc_comment(
                &token,
                conv.span_for(abs_range).unwrap_or(SpanData { range: rel_range, anchor }),
            ) {
                result.extend(tokens);
            }
            continue;
        }
        let tt = if kind.is_punct() && kind != UNDERSCORE {
            let expected = match subtree.delimiter.kind {
                tt::DelimiterKind::Parenthesis => Some(T![')']),
                tt::DelimiterKind::Brace => Some(T!['}']),
                tt::DelimiterKind::Bracket => Some(T![']']),
                tt::DelimiterKind::Invisible => None,
            };

            if let Some(expected) = expected {
                if kind == expected {
                    if let Some(mut subtree) = stack.pop() {
                        subtree.delimiter.close = conv
                            .span_for(abs_range)
                            .unwrap_or(SpanData { range: rel_range, anchor });
                        stack.last_mut().token_trees.push(subtree.into());
                    }
                    continue;
                }
            }

            let delim = match kind {
                T!['('] => Some(tt::DelimiterKind::Parenthesis),
                T!['{'] => Some(tt::DelimiterKind::Brace),
                T!['['] => Some(tt::DelimiterKind::Bracket),
                _ => None,
            };

            if let Some(kind) = delim {
                let subtree = tt::Subtree {
                    delimiter: tt::Delimiter {
                        // FIXME: Open and close spans
                        open: conv
                            .span_for(abs_range)
                            .unwrap_or(SpanData { range: rel_range, anchor }),
                        close: Span::DUMMY,
                        kind,
                    },
                    token_trees: vec![],
                };
                stack.push(subtree);
                continue;
            }

            let spacing = match conv.peek().map(|next| next.kind(conv)) {
                Some(kind) if is_single_token_op(kind) => tt::Spacing::Joint,
                _ => tt::Spacing::Alone,
            };
            let char = match token.to_char(conv) {
                Some(c) => c,
                None => {
                    panic!("Token from lexer must be single char: token = {token:#?}");
                }
            };
            tt::Leaf::from(tt::Punct {
                char,
                spacing,
                span: conv.span_for(abs_range).unwrap_or(SpanData { range: rel_range, anchor }),
            })
            .into()
        } else {
            macro_rules! make_leaf {
                ($i:ident) => {
                    tt::$i {
                        span: conv
                            .span_for(abs_range)
                            .unwrap_or(SpanData { range: rel_range, anchor }),
                        text: token.to_text(conv),
                    }
                    .into()
                };
            }
            let leaf: tt::Leaf<_> = match kind {
                T![true] | T![false] => make_leaf!(Ident),
                IDENT => make_leaf!(Ident),
                UNDERSCORE => make_leaf!(Ident),
                k if k.is_keyword() => make_leaf!(Ident),
                k if k.is_literal() => make_leaf!(Literal),
                // FIXME: Check whether span splitting works as intended
                LIFETIME_IDENT => {
                    let char_unit = TextSize::of('\'');
                    let r = TextRange::at(rel_range.start(), char_unit);
                    let apostrophe = tt::Leaf::from(tt::Punct {
                        char: '\'',
                        spacing: tt::Spacing::Joint,
                        span: conv.span_for(abs_range).unwrap_or(SpanData { range: r, anchor }),
                    });
                    result.push(apostrophe.into());

                    let r =
                        TextRange::at(rel_range.start() + char_unit, rel_range.len() - char_unit);
                    let ident = tt::Leaf::from(tt::Ident {
                        text: SmolStr::new(&token.to_text(conv)[1..]),
                        span: conv.span_for(abs_range).unwrap_or(SpanData { range: r, anchor }),
                    });
                    result.push(ident.into());
                    continue;
                }
                _ => continue,
            };

            leaf.into()
        };
        result.push(tt);
    }

    // If we get here, we've consumed all input tokens.
    // We might have more than one subtree in the stack, if the delimiters are improperly balanced.
    // Merge them so we're left with one.
    while let Some(entry) = stack.pop() {
        let parent = stack.last_mut();

        let leaf: tt::Leaf<_> = tt::Punct {
            span: entry.delimiter.open,
            char: match entry.delimiter.kind {
                tt::DelimiterKind::Parenthesis => '(',
                tt::DelimiterKind::Brace => '{',
                tt::DelimiterKind::Bracket => '[',
                tt::DelimiterKind::Invisible => '$',
            },
            spacing: tt::Spacing::Alone,
        }
        .into();
        parent.token_trees.push(leaf.into());
        parent.token_trees.extend(entry.token_trees);
    }

    let subtree = stack.into_last();
    if let [tt::TokenTree::Subtree(first)] = &*subtree.token_trees {
        first.clone()
    } else {
        subtree
    }
}

fn is_single_token_op(kind: SyntaxKind) -> bool {
    matches!(
        kind,
        EQ | L_ANGLE
            | R_ANGLE
            | BANG
            | AMP
            | PIPE
            | TILDE
            | AT
            | DOT
            | COMMA
            | SEMICOLON
            | COLON
            | POUND
            | DOLLAR
            | QUESTION
            | PLUS
            | MINUS
            | STAR
            | SLASH
            | PERCENT
            | CARET
            // LIFETIME_IDENT will be split into a sequence of `'` (a single quote) and an
            // identifier.
            | LIFETIME_IDENT
    )
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
    let text = format!("\"{}\"", text.escape_debug());
    text.into()
}

fn convert_doc_comment<S: Copy>(
    token: &syntax::SyntaxToken,
    span: S,
) -> Option<Vec<tt::TokenTree<S>>> {
    cov_mark::hit!(test_meta_doc_comments);
    let comment = ast::Comment::cast(token.clone())?;
    let doc = comment.kind().doc?;

    let mk_ident =
        |s: &str| tt::TokenTree::from(tt::Leaf::from(tt::Ident { text: s.into(), span }));

    let mk_punct = |c: char| {
        tt::TokenTree::from(tt::Leaf::from(tt::Punct {
            char: c,
            spacing: tt::Spacing::Alone,
            span,
        }))
    };

    let mk_doc_literal = |comment: &ast::Comment| {
        let lit = tt::Literal { text: doc_comment_text(comment), span };

        tt::TokenTree::from(tt::Leaf::from(lit))
    };

    // Make `doc="\" Comments\""
    let meta_tkns = vec![mk_ident("doc"), mk_punct('='), mk_doc_literal(&comment)];

    // Make `#![]`
    let mut token_trees = Vec::with_capacity(3);
    token_trees.push(mk_punct('#'));
    if let ast::CommentPlacement::Inner = doc {
        token_trees.push(mk_punct('!'));
    }
    token_trees.push(tt::TokenTree::from(tt::Subtree {
        delimiter: tt::Delimiter { open: span, close: span, kind: tt::DelimiterKind::Bracket },
        token_trees: meta_tkns,
    }));

    Some(token_trees)
}

/// A raw token (straight from lexer) converter
struct RawConverter<'a, SpanAnchor> {
    lexed: parser::LexedStr<'a>,
    pos: usize,
    _offset: TextSize,
    file_id: SpanAnchor,
}

trait SrcToken<Ctx>: std::fmt::Debug {
    fn kind(&self, ctx: &Ctx) -> SyntaxKind;

    fn to_char(&self, ctx: &Ctx) -> Option<char>;

    fn to_text(&self, ctx: &Ctx) -> SmolStr;
}

trait TokenConverter<SpanAnchor>: Sized {
    type Token: SrcToken<Self>;

    fn convert_doc_comment(
        &self,
        token: &Self::Token,
        span: SpanData<SpanAnchor>,
    ) -> Option<Vec<tt::TokenTree<SpanData<SpanAnchor>>>>;

    fn bump(&mut self) -> Option<(Self::Token, TextRange, TextRange)>;

    fn peek(&self) -> Option<Self::Token>;

    fn anchor(&self) -> SpanAnchor;
    fn span_for(&self, range: TextRange) -> Option<SpanData<SpanAnchor>>;
}

impl<SpanAnchor> SrcToken<RawConverter<'_, SpanAnchor>> for usize {
    fn kind(&self, ctx: &RawConverter<'_, SpanAnchor>) -> SyntaxKind {
        ctx.lexed.kind(*self)
    }

    fn to_char(&self, ctx: &RawConverter<'_, SpanAnchor>) -> Option<char> {
        ctx.lexed.text(*self).chars().next()
    }

    fn to_text(&self, ctx: &RawConverter<'_, SpanAnchor>) -> SmolStr {
        ctx.lexed.text(*self).into()
    }
}

impl<SpanAnchor: Copy> TokenConverter<SpanAnchor> for RawConverter<'_, SpanAnchor>
where
    SpanData<SpanAnchor>: Span,
{
    type Token = usize;

    fn convert_doc_comment(
        &self,
        &token: &usize,
        span: SpanData<SpanAnchor>,
    ) -> Option<Vec<tt::TokenTree<SpanData<SpanAnchor>>>> {
        let text = self.lexed.text(token);
        convert_doc_comment(&doc_comment(text), span)
    }

    fn bump(&mut self) -> Option<(Self::Token, TextRange, TextRange)> {
        if self.pos == self.lexed.len() {
            return None;
        }
        let token = self.pos;
        self.pos += 1;
        let range = self.lexed.text_range(token);
        let range = TextRange::new(range.start.try_into().ok()?, range.end.try_into().ok()?);

        Some((token, range, range))
    }

    fn peek(&self) -> Option<Self::Token> {
        if self.pos == self.lexed.len() {
            return None;
        }
        Some(self.pos)
    }

    fn anchor(&self) -> SpanAnchor {
        self.file_id
    }
    fn span_for(&self, _: TextRange) -> Option<SpanData<SpanAnchor>> {
        None
    }
}

struct Converter<'a, SpanAnchor> {
    current: Option<SyntaxToken>,
    preorder: PreorderWithTokens,
    range: TextRange,
    punct_offset: Option<(SyntaxToken, TextSize)>,
    /// Used to make the emitted text ranges in the spans relative to the span anchor.
    offset: TextSize,
    file_id: SpanAnchor,
    map: &'a TokenMap<SpanData<SpanAnchor>>,
    censored: Vec<SyntaxNode>,
}

impl<'a, SpanAnchor> Converter<'a, SpanAnchor> {
    fn new(
        node: &SyntaxNode,
        anchor_offset: TextSize,
        file_id: SpanAnchor,
        censored: Vec<SyntaxNode>,
        map: &'a TokenMap<SpanData<SpanAnchor>>,
    ) -> Converter<'a, SpanAnchor> {
        let range = node.text_range();
        let mut preorder = node.preorder_with_tokens();
        let first = Self::next_token(&mut preorder, &censored);
        Converter {
            current: first,
            preorder,
            range,
            punct_offset: None,
            offset: anchor_offset,
            file_id,
            censored,
            map,
        }
    }

    fn next_token(preorder: &mut PreorderWithTokens, censor: &[SyntaxNode]) -> Option<SyntaxToken> {
        while let Some(ev) = preorder.next() {
            match ev {
                WalkEvent::Enter(SyntaxElement::Token(t)) => return Some(t),
                WalkEvent::Enter(SyntaxElement::Node(n)) if censor.contains(&n) => {
                    preorder.skip_subtree()
                }
                _ => (),
            }
        }
        None
    }
}

#[derive(Debug)]
enum SynToken {
    Ordinary(SyntaxToken),
    // FIXME is this supposed to be `Punct`?
    Punct(SyntaxToken, usize),
}

impl SynToken {
    fn token(&self) -> &SyntaxToken {
        match self {
            SynToken::Ordinary(it) | SynToken::Punct(it, _) => it,
        }
    }
}

impl<SpanAnchor> SrcToken<Converter<'_, SpanAnchor>> for SynToken {
    fn kind(&self, ctx: &Converter<'_, SpanAnchor>) -> SyntaxKind {
        match self {
            SynToken::Ordinary(token) => token.kind(),
            SynToken::Punct(..) => SyntaxKind::from_char(self.to_char(ctx).unwrap()).unwrap(),
        }
    }
    fn to_char(&self, _ctx: &Converter<'_, SpanAnchor>) -> Option<char> {
        match self {
            SynToken::Ordinary(_) => None,
            SynToken::Punct(it, i) => it.text().chars().nth(*i),
        }
    }
    fn to_text(&self, _ctx: &Converter<'_, SpanAnchor>) -> SmolStr {
        match self {
            SynToken::Ordinary(token) | SynToken::Punct(token, _) => token.text().into(),
        }
    }
}

impl<SpanAnchor: Copy> TokenConverter<SpanAnchor> for Converter<'_, SpanAnchor>
where
    SpanData<SpanAnchor>: Span,
{
    type Token = SynToken;
    fn convert_doc_comment(
        &self,
        token: &Self::Token,
        span: SpanData<SpanAnchor>,
    ) -> Option<Vec<tt::TokenTree<SpanData<SpanAnchor>>>> {
        convert_doc_comment(token.token(), span)
    }

    fn bump(&mut self) -> Option<(Self::Token, TextRange, TextRange)> {
        if let Some((punct, offset)) = self.punct_offset.clone() {
            if usize::from(offset) + 1 < punct.text().len() {
                let offset = offset + TextSize::of('.');
                let range = punct.text_range();
                self.punct_offset = Some((punct.clone(), offset));
                let range = TextRange::at(range.start() + offset, TextSize::of('.'));
                return Some((
                    SynToken::Punct(punct, u32::from(offset) as usize),
                    range - self.offset,
                    range,
                ));
            }
        }

        let curr = self.current.clone()?;
        if !self.range.contains_range(curr.text_range()) {
            return None;
        }
        self.current = Self::next_token(&mut self.preorder, &self.censored);
        let token = if curr.kind().is_punct() {
            self.punct_offset = Some((curr.clone(), 0.into()));
            let range = curr.text_range();
            let range = TextRange::at(range.start(), TextSize::of('.'));
            (SynToken::Punct(curr, 0 as usize), range - self.offset, range)
        } else {
            self.punct_offset = None;
            let range = curr.text_range();
            (SynToken::Ordinary(curr), range - self.offset, range)
        };

        Some(token)
    }

    fn peek(&self) -> Option<Self::Token> {
        if let Some((punct, mut offset)) = self.punct_offset.clone() {
            offset += TextSize::of('.');
            if usize::from(offset) < punct.text().len() {
                return Some(SynToken::Punct(punct, usize::from(offset)));
            }
        }

        let curr = self.current.clone()?;
        if !self.range.contains_range(curr.text_range()) {
            return None;
        }

        let token = if curr.kind().is_punct() {
            SynToken::Punct(curr, 0 as usize)
        } else {
            SynToken::Ordinary(curr)
        };
        Some(token)
    }

    fn anchor(&self) -> SpanAnchor {
        self.file_id
    }
    fn span_for(&self, range: TextRange) -> Option<SpanData<SpanAnchor>> {
        self.map.span_for_range(range)
    }
}

struct TtTreeSink<'a, SpanAnchor> {
    buf: String,
    cursor: Cursor<'a, SpanData<SpanAnchor>>,
    text_pos: TextSize,
    inner: SyntaxTreeBuilder,
    token_map: TokenMap<SpanData<SpanAnchor>>,
}

impl<'a, SpanAnchor> TtTreeSink<'a, SpanAnchor>
where
    SpanData<SpanAnchor>: Span,
{
    fn new(cursor: Cursor<'a, SpanData<SpanAnchor>>) -> Self {
        TtTreeSink {
            buf: String::new(),
            cursor,
            text_pos: 0.into(),
            inner: SyntaxTreeBuilder::default(),
            token_map: TokenMap::default(),
        }
    }

    fn finish(mut self) -> (Parse<SyntaxNode>, TokenMap<SpanData<SpanAnchor>>) {
        self.token_map.shrink_to_fit();
        (self.inner.finish(), self.token_map)
    }
}

fn delim_to_str(d: tt::DelimiterKind, closing: bool) -> Option<&'static str> {
    let texts = match d {
        tt::DelimiterKind::Parenthesis => "()",
        tt::DelimiterKind::Brace => "{}",
        tt::DelimiterKind::Bracket => "[]",
        tt::DelimiterKind::Invisible => return None,
    };

    let idx = closing as usize;
    Some(&texts[idx..texts.len() - (1 - idx)])
}

impl<SpanAnchor> TtTreeSink<'_, SpanAnchor>
where
    SpanData<SpanAnchor>: Span,
{
    /// Parses a float literal as if it was a one to two name ref nodes with a dot inbetween.
    /// This occurs when a float literal is used as a field access.
    fn float_split(&mut self, has_pseudo_dot: bool) {
        let (text, _span) = match self.cursor.token_tree() {
            Some(tt::buffer::TokenTreeRef::Leaf(tt::Leaf::Literal(lit), _)) => {
                (lit.text.as_str(), lit.span)
            }
            _ => unreachable!(),
        };
        match text.split_once('.') {
            Some((left, right)) => {
                assert!(!left.is_empty());
                self.inner.start_node(SyntaxKind::NAME_REF);
                self.inner.token(SyntaxKind::INT_NUMBER, left);
                self.inner.finish_node();

                // here we move the exit up, the original exit has been deleted in process
                self.inner.finish_node();

                self.inner.token(SyntaxKind::DOT, ".");

                if has_pseudo_dot {
                    assert!(right.is_empty(), "{left}.{right}");
                } else {
                    assert!(!right.is_empty(), "{left}.{right}");
                    self.inner.start_node(SyntaxKind::NAME_REF);
                    self.inner.token(SyntaxKind::INT_NUMBER, right);
                    self.inner.finish_node();

                    // the parser creates an unbalanced start node, we are required to close it here
                    self.inner.finish_node();
                }
            }
            None => unreachable!(),
        }
        self.cursor = self.cursor.bump();
    }

    fn token(&mut self, kind: SyntaxKind, mut n_tokens: u8) {
        if kind == LIFETIME_IDENT {
            n_tokens = 2;
        }

        let mut last = self.cursor;
        for _ in 0..n_tokens {
            let tmp: u8;
            if self.cursor.eof() {
                break;
            }
            last = self.cursor;
            let text: &str = loop {
                break match self.cursor.token_tree() {
                    Some(tt::buffer::TokenTreeRef::Leaf(leaf, _)) => {
                        // Mark the range if needed
                        let (text, span) = match leaf {
                            tt::Leaf::Ident(ident) => (ident.text.as_str(), ident.span),
                            tt::Leaf::Punct(punct) => {
                                assert!(punct.char.is_ascii());
                                tmp = punct.char as u8;
                                (
                                    std::str::from_utf8(std::slice::from_ref(&tmp)).unwrap(),
                                    punct.span,
                                )
                            }
                            tt::Leaf::Literal(lit) => (lit.text.as_str(), lit.span),
                        };
                        let range = TextRange::at(self.text_pos, TextSize::of(text));
                        self.token_map.insert(range, span);
                        self.cursor = self.cursor.bump();
                        text
                    }
                    Some(tt::buffer::TokenTreeRef::Subtree(subtree, _)) => {
                        self.cursor = self.cursor.subtree().unwrap();
                        match delim_to_str(subtree.delimiter.kind, false) {
                            Some(it) => {
                                let range = TextRange::at(self.text_pos, TextSize::of(it));
                                self.token_map.insert(range, subtree.delimiter.open);
                                it
                            }
                            None => continue,
                        }
                    }
                    None => {
                        let parent = self.cursor.end().unwrap();
                        self.cursor = self.cursor.bump();
                        match delim_to_str(parent.delimiter.kind, true) {
                            Some(it) => {
                                let range = TextRange::at(self.text_pos, TextSize::of(it));
                                self.token_map.insert(range, parent.delimiter.close);
                                it
                            }
                            None => continue,
                        }
                    }
                };
            };
            self.buf += text;
            self.text_pos += TextSize::of(text);
        }

        self.inner.token(kind, self.buf.as_str());
        self.buf.clear();
        // Add whitespace between adjoint puncts
        let next = last.bump();
        if let (
            Some(tt::buffer::TokenTreeRef::Leaf(tt::Leaf::Punct(curr), _)),
            Some(tt::buffer::TokenTreeRef::Leaf(tt::Leaf::Punct(next), _)),
        ) = (last.token_tree(), next.token_tree())
        {
            // Note: We always assume the semi-colon would be the last token in
            // other parts of RA such that we don't add whitespace here.
            //
            // When `next` is a `Punct` of `'`, that's a part of a lifetime identifier so we don't
            // need to add whitespace either.
            if curr.spacing == tt::Spacing::Alone && curr.char != ';' && next.char != '\'' {
                self.inner.token(WHITESPACE, " ");
                self.text_pos += TextSize::of(' ');
            }
        }
    }

    fn start_node(&mut self, kind: SyntaxKind) {
        self.inner.start_node(kind);
    }

    fn finish_node(&mut self) {
        self.inner.finish_node();
    }

    fn error(&mut self, error: String) {
        self.inner.error(error, self.text_pos)
    }
}
