//! Conversions between [`SyntaxNode`] and [`tt::TokenTree`].

#![cfg_attr(feature = "in-rust-tree", feature(rustc_private))]

#[cfg(feature = "in-rust-tree")]
extern crate rustc_driver as _;

use std::{collections::VecDeque, fmt, hash::Hash};

use intern::Symbol;
use rustc_hash::{FxHashMap, FxHashSet};
use span::{Edition, Span, SpanAnchor, SpanMap, SyntaxContext};
use stdx::{format_to, never};
use syntax::{
    AstToken, Parse, PreorderWithTokens, SmolStr, SyntaxElement,
    SyntaxKind::{self, *},
    SyntaxNode, SyntaxToken, SyntaxTreeBuilder, T, TextRange, TextSize, WalkEvent,
    ast::{self, make::tokens::doc_comment},
    format_smolstr,
};
use tt::{Punct, buffer::Cursor, token_to_literal};

pub mod prettify_macro_expansion;
mod to_parser_input;
pub use to_parser_input::to_parser_input;
// FIXME: we probably should re-think  `token_tree_to_syntax_node` interfaces
pub use ::parser::TopEntryPoint;

#[cfg(test)]
mod tests;

pub trait SpanMapper {
    fn span_for(&self, range: TextRange) -> Span;
}

impl SpanMapper for SpanMap {
    fn span_for(&self, range: TextRange) -> Span {
        self.span_at(range.start())
    }
}

impl<SM: SpanMapper> SpanMapper for &SM {
    fn span_for(&self, range: TextRange) -> Span {
        SM::span_for(self, range)
    }
}

/// Dummy things for testing where spans don't matter.
pub mod dummy_test_span_utils {

    use span::{Span, SyntaxContext};

    use super::*;

    pub const DUMMY: Span = Span {
        range: TextRange::empty(TextSize::new(0)),
        anchor: span::SpanAnchor {
            file_id: span::EditionedFileId::new(
                span::FileId::from_raw(0xe4e4e),
                span::Edition::CURRENT,
            ),
            ast_id: span::ROOT_ERASED_FILE_AST_ID,
        },
        ctx: SyntaxContext::root(Edition::CURRENT),
    };

    pub struct DummyTestSpanMap;

    impl SpanMapper for DummyTestSpanMap {
        fn span_for(&self, range: syntax::TextRange) -> Span {
            Span {
                range,
                anchor: span::SpanAnchor {
                    file_id: span::EditionedFileId::new(
                        span::FileId::from_raw(0xe4e4e),
                        span::Edition::CURRENT,
                    ),
                    ast_id: span::ROOT_ERASED_FILE_AST_ID,
                },
                ctx: SyntaxContext::root(Edition::CURRENT),
            }
        }
    }
}

/// Doc comment desugaring differs between mbe and proc-macros.
#[derive(Copy, Clone, PartialEq, Eq)]
pub enum DocCommentDesugarMode {
    /// Desugars doc comments as quoted raw strings
    Mbe,
    /// Desugars doc comments as quoted strings
    ProcMacro,
}

/// Converts a syntax tree to a [`tt::Subtree`] using the provided span map to populate the
/// subtree's spans.
pub fn syntax_node_to_token_tree<SpanMap>(
    node: &SyntaxNode,
    map: SpanMap,
    span: Span,
    mode: DocCommentDesugarMode,
) -> tt::TopSubtree
where
    SpanMap: SpanMapper,
{
    let mut c =
        Converter::new(node, map, Default::default(), Default::default(), span, mode, |_, _| {
            (true, Vec::new())
        });
    convert_tokens(&mut c)
}

/// Converts a syntax tree to a [`tt::Subtree`] using the provided span map to populate the
/// subtree's spans. Additionally using the append and remove parameters, the additional tokens can
/// be injected or hidden from the output.
pub fn syntax_node_to_token_tree_modified<SpanMap, OnEvent>(
    node: &SyntaxNode,
    map: SpanMap,
    append: FxHashMap<SyntaxElement, Vec<tt::Leaf>>,
    remove: FxHashSet<SyntaxElement>,
    call_site: Span,
    mode: DocCommentDesugarMode,
    on_enter: OnEvent,
) -> tt::TopSubtree
where
    SpanMap: SpanMapper,
    OnEvent: FnMut(&mut PreorderWithTokens, &WalkEvent<SyntaxElement>) -> (bool, Vec<tt::Leaf>),
{
    let mut c = Converter::new(node, map, append, remove, call_site, mode, on_enter);
    convert_tokens(&mut c)
}

// The following items are what `rustc` macro can be parsed into :
// link: https://github.com/rust-lang/rust/blob/9ebf47851a357faa4cd97f4b1dc7835f6376e639/src/libsyntax/ext/expand.rs#L141
// * Expr(Box<ast::Expr>)                     -> token_tree_to_expr
// * Pat(Box<ast::Pat>)                       -> token_tree_to_pat
// * Ty(Box<ast::Ty>)                         -> token_tree_to_ty
// * Stmts(SmallVec<[ast::Stmt; 1]>)        -> token_tree_to_stmts
// * Items(SmallVec<[Box<ast::Item>; 1]>)     -> token_tree_to_items
//
// * TraitItems(SmallVec<[ast::TraitItem; 1]>)
// * AssocItems(SmallVec<[ast::AssocItem; 1]>)
// * ForeignItems(SmallVec<[ast::ForeignItem; 1]>

/// Converts a [`tt::Subtree`] back to a [`SyntaxNode`].
/// The produced `SpanMap` contains a mapping from the syntax nodes offsets to the subtree's spans.
pub fn token_tree_to_syntax_node(
    tt: &tt::TopSubtree,
    entry_point: parser::TopEntryPoint,
    span_to_edition: &mut dyn FnMut(SyntaxContext) -> Edition,
) -> (Parse<SyntaxNode>, SpanMap)
where
    SyntaxContext: Copy + fmt::Debug + PartialEq + PartialEq + Eq + Hash,
{
    let buffer = tt.view().strip_invisible();
    let parser_input = to_parser_input(buffer, span_to_edition);
    // It matters what edition we parse with even when we escape all identifiers correctly.
    let parser_output = entry_point.parse(&parser_input);
    let mut tree_sink = TtTreeSink::new(buffer.cursor());
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
            parser::Step::Error { msg } => tree_sink.error(msg.to_owned()),
        }
    }
    tree_sink.finish()
}

/// Convert a string to a `TokenTree`. The spans of the subtree will be anchored to the provided
/// anchor with the given context.
pub fn parse_to_token_tree(
    edition: Edition,
    anchor: SpanAnchor,
    ctx: SyntaxContext,
    text: &str,
) -> Option<tt::TopSubtree> {
    let lexed = parser::LexedStr::new(edition, text);
    if lexed.errors().next().is_some() {
        return None;
    }
    let mut conv =
        RawConverter { lexed, anchor, pos: 0, ctx, mode: DocCommentDesugarMode::ProcMacro };
    Some(convert_tokens(&mut conv))
}

/// Convert a string to a `TokenTree`. The passed span will be used for all spans of the produced subtree.
pub fn parse_to_token_tree_static_span(
    edition: Edition,
    span: Span,
    text: &str,
) -> Option<tt::TopSubtree> {
    let lexed = parser::LexedStr::new(edition, text);
    if lexed.errors().next().is_some() {
        return None;
    }
    let mut conv =
        StaticRawConverter { lexed, pos: 0, span, mode: DocCommentDesugarMode::ProcMacro };
    Some(convert_tokens(&mut conv))
}

fn convert_tokens<C>(conv: &mut C) -> tt::TopSubtree
where
    C: TokenConverter,
    C::Token: fmt::Debug,
{
    let mut builder =
        tt::TopSubtreeBuilder::new(tt::Delimiter::invisible_spanned(conv.call_site()));

    while let Some((token, abs_range)) = conv.bump() {
        let tt = match token.as_leaf() {
            // These delimiters are not actually valid punctuation, but we produce them in syntax fixup.
            // So we need to handle them specially here.
            Some(&tt::Leaf::Punct(Punct {
                char: char @ ('(' | ')' | '{' | '}' | '[' | ']'),
                span,
                spacing: _,
            })) => {
                let found_expected_delimiter =
                    builder.expected_delimiters().enumerate().find(|(_, delim)| match delim {
                        tt::DelimiterKind::Parenthesis => char == ')',
                        tt::DelimiterKind::Brace => char == '}',
                        tt::DelimiterKind::Bracket => char == ']',
                        tt::DelimiterKind::Invisible => false,
                    });
                if let Some((idx, _)) = found_expected_delimiter {
                    for _ in 0..=idx {
                        builder.close(span);
                    }
                    continue;
                }

                let delim = match char {
                    '(' => tt::DelimiterKind::Parenthesis,
                    '{' => tt::DelimiterKind::Brace,
                    '[' => tt::DelimiterKind::Bracket,
                    _ => panic!("unmatched closing delimiter from syntax fixup"),
                };

                // Start a new subtree
                builder.open(delim, span);
                continue;
            }
            Some(leaf) => leaf.clone(),
            None => match token.kind(conv) {
                // Desugar doc comments into doc attributes
                COMMENT => {
                    let span = conv.span_for(abs_range);
                    conv.convert_doc_comment(&token, span, &mut builder);
                    continue;
                }
                kind if kind.is_punct() && kind != UNDERSCORE => {
                    let found_expected_delimiter =
                        builder.expected_delimiters().enumerate().find(|(_, delim)| match delim {
                            tt::DelimiterKind::Parenthesis => kind == T![')'],
                            tt::DelimiterKind::Brace => kind == T!['}'],
                            tt::DelimiterKind::Bracket => kind == T![']'],
                            tt::DelimiterKind::Invisible => false,
                        });

                    // Current token is a closing delimiter that we expect, fix up the closing span
                    // and end the subtree here.
                    // We also close any open inner subtrees that might be missing their delimiter.
                    if let Some((idx, _)) = found_expected_delimiter {
                        for _ in 0..=idx {
                            // FIXME: record an error somewhere if we're closing more than one tree here?
                            builder.close(conv.span_for(abs_range));
                        }
                        continue;
                    }

                    let delim = match kind {
                        T!['('] => Some(tt::DelimiterKind::Parenthesis),
                        T!['{'] => Some(tt::DelimiterKind::Brace),
                        T!['['] => Some(tt::DelimiterKind::Bracket),
                        _ => None,
                    };

                    // Start a new subtree
                    if let Some(kind) = delim {
                        builder.open(kind, conv.span_for(abs_range));
                        continue;
                    }

                    let spacing = match conv.peek().map(|next| next.kind(conv)) {
                        Some(kind) if is_single_token_op(kind) => tt::Spacing::Joint,
                        _ => tt::Spacing::Alone,
                    };
                    let Some(char) = token.to_char(conv) else {
                        panic!("Token from lexer must be single char: token = {token:#?}")
                    };
                    // FIXME: this might still be an unmatched closing delimiter? Maybe we should assert here
                    tt::Leaf::from(tt::Punct { char, spacing, span: conv.span_for(abs_range) })
                }
                kind => {
                    macro_rules! make_ident {
                        () => {
                            tt::Ident {
                                span: conv.span_for(abs_range),
                                sym: Symbol::intern(&token.to_text(conv)),
                                is_raw: tt::IdentIsRaw::No,
                            }
                            .into()
                        };
                    }
                    let leaf: tt::Leaf = match kind {
                        k if k.is_any_identifier() => {
                            let text = token.to_text(conv);
                            tt::Ident::new(&text, conv.span_for(abs_range)).into()
                        }
                        UNDERSCORE => make_ident!(),
                        k if k.is_literal() => {
                            let text = token.to_text(conv);
                            let span = conv.span_for(abs_range);
                            token_to_literal(&text, span).into()
                        }
                        LIFETIME_IDENT => {
                            let apostrophe = tt::Leaf::from(tt::Punct {
                                char: '\'',
                                spacing: tt::Spacing::Joint,
                                span: conv
                                    .span_for(TextRange::at(abs_range.start(), TextSize::of('\''))),
                            });
                            builder.push(apostrophe);

                            let ident = tt::Leaf::from(tt::Ident {
                                sym: Symbol::intern(&token.to_text(conv)[1..]),
                                span: conv.span_for(TextRange::new(
                                    abs_range.start() + TextSize::of('\''),
                                    abs_range.end(),
                                )),
                                is_raw: tt::IdentIsRaw::No,
                            });
                            builder.push(ident);
                            continue;
                        }
                        _ => continue,
                    };

                    leaf
                }
            },
        };

        builder.push(tt);
    }

    while builder.expected_delimiters().next().is_some() {
        // FIXME: record an error somewhere?
        builder.close(conv.call_site());
    }
    builder.build_skip_top_subtree()
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
///
/// Note that proc-macros desugar with string literals where as macro_rules macros desugar with raw string literals.
pub fn desugar_doc_comment_text(text: &str, mode: DocCommentDesugarMode) -> (Symbol, tt::LitKind) {
    match mode {
        DocCommentDesugarMode::Mbe => {
            let mut num_of_hashes = 0;
            let mut count = 0;
            for ch in text.chars() {
                count = match ch {
                    '"' => 1,
                    '#' if count > 0 => count + 1,
                    _ => 0,
                };
                num_of_hashes = num_of_hashes.max(count);
            }

            // Quote raw string with delimiters
            (Symbol::intern(text), tt::LitKind::StrRaw(num_of_hashes))
        }
        // Quote string with delimiters
        DocCommentDesugarMode::ProcMacro => {
            (Symbol::intern(&format_smolstr!("{}", text.escape_debug())), tt::LitKind::Str)
        }
    }
}

fn convert_doc_comment(
    token: &syntax::SyntaxToken,
    span: Span,
    mode: DocCommentDesugarMode,
    builder: &mut tt::TopSubtreeBuilder,
) {
    let Some(comment) = ast::Comment::cast(token.clone()) else { return };
    let Some(doc) = comment.kind().doc else { return };

    let mk_ident = |s: &str| {
        tt::Leaf::from(tt::Ident { sym: Symbol::intern(s), span, is_raw: tt::IdentIsRaw::No })
    };

    let mk_punct =
        |c: char| tt::Leaf::from(tt::Punct { char: c, spacing: tt::Spacing::Alone, span });

    let mk_doc_literal = |comment: &ast::Comment| {
        let prefix_len = comment.prefix().len();
        let mut text = &comment.text()[prefix_len..];

        // Remove ending "*/"
        if comment.kind().shape == ast::CommentShape::Block {
            text = &text[0..text.len() - 2];
        }
        let (text, kind) = desugar_doc_comment_text(text, mode);
        let lit = tt::Literal { text_and_suffix: text, span, kind, suffix_len: 0 };

        tt::Leaf::from(lit)
    };

    // Make `doc="\" Comments\""
    let meta_tkns = [mk_ident("doc"), mk_punct('='), mk_doc_literal(&comment)];

    // Make `#![]`
    builder.push(mk_punct('#'));
    if let ast::CommentPlacement::Inner = doc {
        builder.push(mk_punct('!'));
    }
    builder.open(tt::DelimiterKind::Bracket, span);
    builder.extend(meta_tkns);
    builder.close(span);
}

/// A raw token (straight from lexer) converter
struct RawConverter<'a> {
    lexed: parser::LexedStr<'a>,
    pos: usize,
    anchor: SpanAnchor,
    ctx: SyntaxContext,
    mode: DocCommentDesugarMode,
}
/// A raw token (straight from lexer) converter that gives every token the same span.
struct StaticRawConverter<'a> {
    lexed: parser::LexedStr<'a>,
    pos: usize,
    span: Span,
    mode: DocCommentDesugarMode,
}

trait SrcToken<Ctx> {
    fn kind(&self, ctx: &Ctx) -> SyntaxKind;

    fn to_char(&self, ctx: &Ctx) -> Option<char>;

    fn to_text(&self, ctx: &Ctx) -> SmolStr;

    fn as_leaf(&self) -> Option<&tt::Leaf> {
        None
    }
}

trait TokenConverter: Sized {
    type Token: SrcToken<Self>;

    fn convert_doc_comment(
        &self,
        token: &Self::Token,
        span: Span,
        builder: &mut tt::TopSubtreeBuilder,
    );

    fn bump(&mut self) -> Option<(Self::Token, TextRange)>;

    fn peek(&self) -> Option<Self::Token>;

    fn span_for(&self, range: TextRange) -> Span;

    fn call_site(&self) -> Span;
}

impl SrcToken<RawConverter<'_>> for usize {
    fn kind(&self, ctx: &RawConverter<'_>) -> SyntaxKind {
        ctx.lexed.kind(*self)
    }

    fn to_char(&self, ctx: &RawConverter<'_>) -> Option<char> {
        ctx.lexed.text(*self).chars().next()
    }

    fn to_text(&self, ctx: &RawConverter<'_>) -> SmolStr {
        ctx.lexed.text(*self).into()
    }
}

impl SrcToken<StaticRawConverter<'_>> for usize {
    fn kind(&self, ctx: &StaticRawConverter<'_>) -> SyntaxKind {
        ctx.lexed.kind(*self)
    }

    fn to_char(&self, ctx: &StaticRawConverter<'_>) -> Option<char> {
        ctx.lexed.text(*self).chars().next()
    }

    fn to_text(&self, ctx: &StaticRawConverter<'_>) -> SmolStr {
        ctx.lexed.text(*self).into()
    }
}

impl TokenConverter for RawConverter<'_> {
    type Token = usize;

    fn convert_doc_comment(&self, &token: &usize, span: Span, builder: &mut tt::TopSubtreeBuilder) {
        let text = self.lexed.text(token);
        convert_doc_comment(&doc_comment(text), span, self.mode, builder);
    }

    fn bump(&mut self) -> Option<(Self::Token, TextRange)> {
        if self.pos == self.lexed.len() {
            return None;
        }
        let token = self.pos;
        self.pos += 1;
        let range = self.lexed.text_range(token);
        let range = TextRange::new(range.start.try_into().ok()?, range.end.try_into().ok()?);

        Some((token, range))
    }

    fn peek(&self) -> Option<Self::Token> {
        if self.pos == self.lexed.len() {
            return None;
        }
        Some(self.pos)
    }

    fn span_for(&self, range: TextRange) -> Span {
        Span { range, anchor: self.anchor, ctx: self.ctx }
    }

    fn call_site(&self) -> Span {
        Span { range: TextRange::empty(0.into()), anchor: self.anchor, ctx: self.ctx }
    }
}

impl TokenConverter for StaticRawConverter<'_> {
    type Token = usize;

    fn convert_doc_comment(&self, &token: &usize, span: Span, builder: &mut tt::TopSubtreeBuilder) {
        let text = self.lexed.text(token);
        convert_doc_comment(&doc_comment(text), span, self.mode, builder);
    }

    fn bump(&mut self) -> Option<(Self::Token, TextRange)> {
        if self.pos == self.lexed.len() {
            return None;
        }
        let token = self.pos;
        self.pos += 1;
        let range = self.lexed.text_range(token);
        let range = TextRange::new(range.start.try_into().ok()?, range.end.try_into().ok()?);

        Some((token, range))
    }

    fn peek(&self) -> Option<Self::Token> {
        if self.pos == self.lexed.len() {
            return None;
        }
        Some(self.pos)
    }

    fn span_for(&self, _: TextRange) -> Span {
        self.span
    }

    fn call_site(&self) -> Span {
        self.span
    }
}

struct Converter<SpanMap, OnEvent> {
    current: Option<SyntaxToken>,
    current_leaves: VecDeque<tt::Leaf>,
    preorder: PreorderWithTokens,
    range: TextRange,
    punct_offset: Option<(SyntaxToken, TextSize)>,
    /// Used to make the emitted text ranges in the spans relative to the span anchor.
    map: SpanMap,
    append: FxHashMap<SyntaxElement, Vec<tt::Leaf>>,
    remove: FxHashSet<SyntaxElement>,
    call_site: Span,
    mode: DocCommentDesugarMode,
    on_event: OnEvent,
}

impl<SpanMap, OnEvent> Converter<SpanMap, OnEvent>
where
    OnEvent: FnMut(&mut PreorderWithTokens, &WalkEvent<SyntaxElement>) -> (bool, Vec<tt::Leaf>),
{
    fn new(
        node: &SyntaxNode,
        map: SpanMap,
        append: FxHashMap<SyntaxElement, Vec<tt::Leaf>>,
        remove: FxHashSet<SyntaxElement>,
        call_site: Span,
        mode: DocCommentDesugarMode,
        on_enter: OnEvent,
    ) -> Self {
        let mut converter = Converter {
            current: None,
            preorder: node.preorder_with_tokens(),
            range: node.text_range(),
            punct_offset: None,
            map,
            append,
            remove,
            call_site,
            current_leaves: VecDeque::new(),
            mode,
            on_event: on_enter,
        };
        converter.current = converter.next_token();
        converter
    }

    fn next_token(&mut self) -> Option<SyntaxToken> {
        while let Some(ev) = self.preorder.next() {
            let (keep_event, insert_leaves) = (self.on_event)(&mut self.preorder, &ev);
            self.current_leaves.extend(insert_leaves);
            if !keep_event {
                continue;
            }
            match ev {
                WalkEvent::Enter(token) => {
                    if self.remove.contains(&token) {
                        match token {
                            syntax::NodeOrToken::Token(_) => {
                                continue;
                            }
                            node => {
                                self.preorder.skip_subtree();
                                if let Some(v) = self.append.remove(&node) {
                                    self.current_leaves.extend(v);
                                    continue;
                                }
                            }
                        }
                    } else if let syntax::NodeOrToken::Token(token) = token {
                        return Some(token);
                    }
                }
                WalkEvent::Leave(ele) => {
                    if let Some(v) = self.append.remove(&ele) {
                        self.current_leaves.extend(v);
                        continue;
                    }
                }
            }
        }
        None
    }
}

#[derive(Debug)]
enum SynToken {
    Ordinary(SyntaxToken),
    Punct { token: SyntaxToken, offset: usize },
    Leaf(tt::Leaf),
}

impl SynToken {
    fn token(&self) -> &SyntaxToken {
        match self {
            SynToken::Ordinary(it) | SynToken::Punct { token: it, offset: _ } => it,
            SynToken::Leaf(_) => unreachable!(),
        }
    }
}

impl<SpanMap, OnEvent> SrcToken<Converter<SpanMap, OnEvent>> for SynToken {
    fn kind(&self, _ctx: &Converter<SpanMap, OnEvent>) -> SyntaxKind {
        match self {
            SynToken::Ordinary(token) => token.kind(),
            SynToken::Punct { token, offset: i } => {
                SyntaxKind::from_char(token.text().chars().nth(*i).unwrap()).unwrap()
            }
            SynToken::Leaf(_) => {
                never!();
                SyntaxKind::ERROR
            }
        }
    }
    fn to_char(&self, _ctx: &Converter<SpanMap, OnEvent>) -> Option<char> {
        match self {
            SynToken::Ordinary(_) => None,
            SynToken::Punct { token: it, offset: i } => it.text().chars().nth(*i),
            SynToken::Leaf(_) => None,
        }
    }
    fn to_text(&self, _ctx: &Converter<SpanMap, OnEvent>) -> SmolStr {
        match self {
            SynToken::Ordinary(token) | SynToken::Punct { token, offset: _ } => token.text().into(),
            SynToken::Leaf(_) => {
                never!();
                "".into()
            }
        }
    }
    fn as_leaf(&self) -> Option<&tt::Leaf> {
        match self {
            SynToken::Ordinary(_) | SynToken::Punct { .. } => None,
            SynToken::Leaf(it) => Some(it),
        }
    }
}

impl<SpanMap, OnEvent> TokenConverter for Converter<SpanMap, OnEvent>
where
    SpanMap: SpanMapper,
    OnEvent: FnMut(&mut PreorderWithTokens, &WalkEvent<SyntaxElement>) -> (bool, Vec<tt::Leaf>),
{
    type Token = SynToken;
    fn convert_doc_comment(
        &self,
        token: &Self::Token,
        span: Span,
        builder: &mut tt::TopSubtreeBuilder,
    ) {
        convert_doc_comment(token.token(), span, self.mode, builder);
    }

    fn bump(&mut self) -> Option<(Self::Token, TextRange)> {
        if let Some((punct, offset)) = self.punct_offset.clone()
            && usize::from(offset) + 1 < punct.text().len()
        {
            let offset = offset + TextSize::of('.');
            let range = punct.text_range();
            self.punct_offset = Some((punct.clone(), offset));
            let range = TextRange::at(range.start() + offset, TextSize::of('.'));
            return Some((
                SynToken::Punct { token: punct, offset: u32::from(offset) as usize },
                range,
            ));
        }

        if let Some(leaf) = self.current_leaves.pop_front() {
            return Some((SynToken::Leaf(leaf), TextRange::empty(TextSize::new(0))));
        }

        let curr = self.current.clone()?;
        if !self.range.contains_range(curr.text_range()) {
            return None;
        }

        self.current = self.next_token();
        let token = if curr.kind().is_punct() {
            self.punct_offset = Some((curr.clone(), 0.into()));
            let range = curr.text_range();
            let range = TextRange::at(range.start(), TextSize::of('.'));
            (SynToken::Punct { token: curr, offset: 0_usize }, range)
        } else {
            self.punct_offset = None;
            let range = curr.text_range();
            (SynToken::Ordinary(curr), range)
        };

        Some(token)
    }

    fn peek(&self) -> Option<Self::Token> {
        if let Some((punct, mut offset)) = self.punct_offset.clone() {
            offset += TextSize::of('.');
            if usize::from(offset) < punct.text().len() {
                return Some(SynToken::Punct { token: punct, offset: usize::from(offset) });
            }
        }

        let curr = self.current.clone()?;
        if !self.range.contains_range(curr.text_range()) {
            return None;
        }

        let token = if curr.kind().is_punct() {
            SynToken::Punct { token: curr, offset: 0_usize }
        } else {
            SynToken::Ordinary(curr)
        };
        Some(token)
    }

    fn span_for(&self, range: TextRange) -> Span {
        self.map.span_for(range)
    }
    fn call_site(&self) -> Span {
        self.call_site
    }
}

struct TtTreeSink<'a> {
    buf: String,
    cursor: Cursor<'a>,
    text_pos: TextSize,
    inner: SyntaxTreeBuilder,
    token_map: SpanMap,
}

impl<'a> TtTreeSink<'a> {
    fn new(cursor: Cursor<'a>) -> Self {
        TtTreeSink {
            buf: String::new(),
            cursor,
            text_pos: 0.into(),
            inner: SyntaxTreeBuilder::default(),
            token_map: SpanMap::empty(),
        }
    }

    fn finish(mut self) -> (Parse<SyntaxNode>, SpanMap) {
        self.token_map.finish();
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

impl TtTreeSink<'_> {
    /// Parses a float literal as if it was a one to two name ref nodes with a dot inbetween.
    /// This occurs when a float literal is used as a field access.
    fn float_split(&mut self, has_pseudo_dot: bool) {
        let token_tree = self.cursor.token_tree();
        let (text, span) = match &token_tree {
            Some(tt::TokenTree::Leaf(tt::Leaf::Literal(
                lit @ tt::Literal { span, kind: tt::LitKind::Float, .. },
            ))) => (lit.text(), *span),
            tt => unreachable!("{tt:?}"),
        };
        // FIXME: Span splitting
        match text.split_once('.') {
            Some((left, right)) => {
                assert!(!left.is_empty());

                self.inner.start_node(SyntaxKind::NAME_REF);
                self.inner.token(SyntaxKind::INT_NUMBER, left);
                self.inner.finish_node();
                self.token_map.push(self.text_pos + TextSize::of(left), span);

                // here we move the exit up, the original exit has been deleted in process
                self.inner.finish_node();

                self.inner.token(SyntaxKind::DOT, ".");
                self.token_map.push(self.text_pos + TextSize::of(left) + TextSize::of("."), span);

                if has_pseudo_dot {
                    assert!(right.is_empty(), "{left}.{right}");
                } else {
                    assert!(!right.is_empty(), "{left}.{right}");
                    self.inner.start_node(SyntaxKind::NAME_REF);
                    self.inner.token(SyntaxKind::INT_NUMBER, right);
                    self.token_map.push(self.text_pos + TextSize::of(text), span);
                    self.inner.finish_node();

                    // the parser creates an unbalanced start node, we are required to close it here
                    self.inner.finish_node();
                }
                self.text_pos += TextSize::of(text);
            }
            None => unreachable!(),
        }
        self.cursor.bump();
    }

    fn token(&mut self, kind: SyntaxKind, mut n_tokens: u8) {
        if kind == LIFETIME_IDENT {
            n_tokens = 2;
        }

        let mut last_two = self.cursor.peek_two_leaves();
        let mut combined_span = None;
        'tokens: for _ in 0..n_tokens {
            let tmp: u8;
            if self.cursor.eof() {
                break;
            }
            last_two = self.cursor.peek_two_leaves();
            let (text, span) = loop {
                break match self.cursor.token_tree() {
                    Some(tt::TokenTree::Leaf(leaf)) => match leaf {
                        tt::Leaf::Ident(ident) => {
                            if ident.is_raw.yes() {
                                self.buf.push_str("r#");
                                self.text_pos += TextSize::of("r#");
                            }
                            let text = ident.sym.as_str();
                            self.buf += text;
                            self.text_pos += TextSize::of(text);
                            combined_span = match combined_span {
                                None => Some(ident.span),
                                Some(prev_span) => Some(Self::merge_spans(prev_span, ident.span)),
                            };
                            self.cursor.bump();
                            continue 'tokens;
                        }
                        tt::Leaf::Punct(punct) => {
                            assert!(punct.char.is_ascii());
                            tmp = punct.char as u8;
                            let r = (
                                std::str::from_utf8(std::slice::from_ref(&tmp)).unwrap(),
                                punct.span,
                            );
                            self.cursor.bump();
                            r
                        }
                        tt::Leaf::Literal(lit) => {
                            let buf_l = self.buf.len();
                            format_to!(self.buf, "{lit}");
                            debug_assert_ne!(self.buf.len() - buf_l, 0);
                            self.text_pos += TextSize::new((self.buf.len() - buf_l) as u32);
                            combined_span = match combined_span {
                                None => Some(lit.span),
                                Some(prev_span) => Some(Self::merge_spans(prev_span, lit.span)),
                            };
                            self.cursor.bump();
                            continue 'tokens;
                        }
                    },
                    Some(tt::TokenTree::Subtree(subtree)) => {
                        self.cursor.bump();
                        match delim_to_str(subtree.delimiter.kind, false) {
                            Some(it) => (it, subtree.delimiter.open),
                            None => continue,
                        }
                    }
                    None => {
                        let parent = self.cursor.end();
                        match delim_to_str(parent.delimiter.kind, true) {
                            Some(it) => (it, parent.delimiter.close),
                            None => continue,
                        }
                    }
                };
            };
            self.buf += text;
            self.text_pos += TextSize::of(text);
            combined_span = match combined_span {
                None => Some(span),
                Some(prev_span) => Some(Self::merge_spans(prev_span, span)),
            }
        }

        self.token_map.push(self.text_pos, combined_span.expect("expected at least one token"));
        self.inner.token(kind, self.buf.as_str());
        self.buf.clear();
        // FIXME: Emitting whitespace for this is really just a hack, we should get rid of it.
        // Add whitespace between adjoint puncts
        if let Some([tt::Leaf::Punct(curr), tt::Leaf::Punct(next)]) = last_two {
            // Note: We always assume the semi-colon would be the last token in
            // other parts of RA such that we don't add whitespace here.
            //
            // When `next` is a `Punct` of `'`, that's a part of a lifetime identifier so we don't
            // need to add whitespace either.
            if curr.spacing == tt::Spacing::Alone && curr.char != ';' && next.char != '\'' {
                self.inner.token(WHITESPACE, " ");
                self.text_pos += TextSize::of(' ');
                self.token_map.push(self.text_pos, curr.span);
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

    fn merge_spans(a: Span, b: Span) -> Span {
        // We don't do what rustc does exactly, rustc does something clever when the spans have different syntax contexts
        // but this runs afoul of our separation between `span` and `hir-expand`.
        Span {
            range: if a.ctx == b.ctx && a.anchor == b.anchor {
                TextRange::new(
                    std::cmp::min(a.range.start(), b.range.start()),
                    std::cmp::max(a.range.end(), b.range.end()),
                )
            } else {
                // Combining ranges make no sense when they come from different syntax contexts.
                a.range
            },
            anchor: a.anchor,
            ctx: a.ctx,
        }
    }
}
