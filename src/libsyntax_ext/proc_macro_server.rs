use errors::{Diagnostic, DiagnosticBuilder};

use std::panic;

use proc_macro::bridge::{server, TokenTree};
use proc_macro::{Delimiter, Level, LineColumn, Spacing};

use rustc_data_structures::sync::Lrc;
use std::ascii;
use std::ops::Bound;
use syntax::ast;
use syntax::ext::base::ExtCtxt;
use syntax::parse::lexer::comments;
use syntax::parse::{self, token, ParseSess};
use syntax::tokenstream::{self, DelimSpan, IsJoint::*, TokenStream, TreeAndJoint};
use syntax_pos::hygiene::{SyntaxContext, Transparency};
use syntax_pos::symbol::{kw, sym, Symbol};
use syntax_pos::{BytePos, FileName, MultiSpan, Pos, SourceFile, Span};

trait FromInternal<T> {
    fn from_internal(x: T) -> Self;
}

trait ToInternal<T> {
    fn to_internal(self) -> T;
}

impl FromInternal<token::DelimToken> for Delimiter {
    fn from_internal(delim: token::DelimToken) -> Delimiter {
        match delim {
            token::Paren => Delimiter::Parenthesis,
            token::Brace => Delimiter::Brace,
            token::Bracket => Delimiter::Bracket,
            token::NoDelim => Delimiter::None,
        }
    }
}

impl ToInternal<token::DelimToken> for Delimiter {
    fn to_internal(self) -> token::DelimToken {
        match self {
            Delimiter::Parenthesis => token::Paren,
            Delimiter::Brace => token::Brace,
            Delimiter::Bracket => token::Bracket,
            Delimiter::None => token::NoDelim,
        }
    }
}

impl FromInternal<(TreeAndJoint, &'_ ParseSess, &'_ mut Vec<Self>)>
    for TokenTree<Group, Punct, Ident, Literal>
{
    fn from_internal(((tree, is_joint), sess, stack): (TreeAndJoint, &ParseSess, &mut Vec<Self>))
                    -> Self {
        use syntax::parse::token::*;

        let joint = is_joint == Joint;
        let Token { kind, span } = match tree {
            tokenstream::TokenTree::Delimited(span, delim, tts) => {
                let delimiter = Delimiter::from_internal(delim);
                return TokenTree::Group(Group {
                    delimiter,
                    stream: tts.into(),
                    span,
                });
            }
            tokenstream::TokenTree::Token(token) => token,
        };

        macro_rules! tt {
            ($ty:ident { $($field:ident $(: $value:expr)*),+ $(,)? }) => (
                TokenTree::$ty(self::$ty {
                    $($field $(: $value)*,)+
                    span,
                })
            );
            ($ty:ident::$method:ident($($value:expr),*)) => (
                TokenTree::$ty(self::$ty::$method($($value,)* span))
            );
        }
        macro_rules! op {
            ($a:expr) => {
                tt!(Punct::new($a, joint))
            };
            ($a:expr, $b:expr) => {{
                stack.push(tt!(Punct::new($b, joint)));
                tt!(Punct::new($a, true))
            }};
            ($a:expr, $b:expr, $c:expr) => {{
                stack.push(tt!(Punct::new($c, joint)));
                stack.push(tt!(Punct::new($b, true)));
                tt!(Punct::new($a, true))
            }};
        }

        match kind {
            Eq => op!('='),
            Lt => op!('<'),
            Le => op!('<', '='),
            EqEq => op!('=', '='),
            Ne => op!('!', '='),
            Ge => op!('>', '='),
            Gt => op!('>'),
            AndAnd => op!('&', '&'),
            OrOr => op!('|', '|'),
            Not => op!('!'),
            Tilde => op!('~'),
            BinOp(Plus) => op!('+'),
            BinOp(Minus) => op!('-'),
            BinOp(Star) => op!('*'),
            BinOp(Slash) => op!('/'),
            BinOp(Percent) => op!('%'),
            BinOp(Caret) => op!('^'),
            BinOp(And) => op!('&'),
            BinOp(Or) => op!('|'),
            BinOp(Shl) => op!('<', '<'),
            BinOp(Shr) => op!('>', '>'),
            BinOpEq(Plus) => op!('+', '='),
            BinOpEq(Minus) => op!('-', '='),
            BinOpEq(Star) => op!('*', '='),
            BinOpEq(Slash) => op!('/', '='),
            BinOpEq(Percent) => op!('%', '='),
            BinOpEq(Caret) => op!('^', '='),
            BinOpEq(And) => op!('&', '='),
            BinOpEq(Or) => op!('|', '='),
            BinOpEq(Shl) => op!('<', '<', '='),
            BinOpEq(Shr) => op!('>', '>', '='),
            At => op!('@'),
            Dot => op!('.'),
            DotDot => op!('.', '.'),
            DotDotDot => op!('.', '.', '.'),
            DotDotEq => op!('.', '.', '='),
            Comma => op!(','),
            Semi => op!(';'),
            Colon => op!(':'),
            ModSep => op!(':', ':'),
            RArrow => op!('-', '>'),
            LArrow => op!('<', '-'),
            FatArrow => op!('=', '>'),
            Pound => op!('#'),
            Dollar => op!('$'),
            Question => op!('?'),
            SingleQuote => op!('\''),

            Ident(name, false) if name == kw::DollarCrate => tt!(Ident::dollar_crate()),
            Ident(name, is_raw) => tt!(Ident::new(name, is_raw)),
            Lifetime(name) => {
                let ident = ast::Ident::new(name, span).without_first_quote();
                stack.push(tt!(Ident::new(ident.name, false)));
                tt!(Punct::new('\'', true))
            }
            Literal(lit) => tt!(Literal { lit }),
            DocComment(c) => {
                let style = comments::doc_comment_style(&c.as_str());
                let stripped = comments::strip_doc_comment_decoration(&c.as_str());
                let mut escaped = String::new();
                for ch in stripped.chars() {
                    escaped.extend(ch.escape_debug());
                }
                let stream = vec![
                    Ident(sym::doc, false),
                    Eq,
                    TokenKind::lit(token::Str, Symbol::intern(&escaped), None),
                ]
                .into_iter()
                .map(|kind| tokenstream::TokenTree::token(kind, span))
                .collect();
                stack.push(TokenTree::Group(Group {
                    delimiter: Delimiter::Bracket,
                    stream,
                    span: DelimSpan::from_single(span),
                }));
                if style == ast::AttrStyle::Inner {
                    stack.push(tt!(Punct::new('!', false)));
                }
                tt!(Punct::new('#', false))
            }

            Interpolated(nt) => {
                let stream = nt.to_tokenstream(sess, span);
                TokenTree::Group(Group {
                    delimiter: Delimiter::None,
                    stream,
                    span: DelimSpan::from_single(span),
                })
            }

            OpenDelim(..) | CloseDelim(..) => unreachable!(),
            Whitespace | Comment | Shebang(..) | Eof => unreachable!(),
        }
    }
}

impl ToInternal<TokenStream> for TokenTree<Group, Punct, Ident, Literal> {
    fn to_internal(self) -> TokenStream {
        use syntax::parse::token::*;

        let (ch, joint, span) = match self {
            TokenTree::Punct(Punct { ch, joint, span }) => (ch, joint, span),
            TokenTree::Group(Group {
                delimiter,
                stream,
                span,
            }) => {
                return tokenstream::TokenTree::Delimited(
                    span,
                    delimiter.to_internal(),
                    stream.into(),
                )
                .into();
            }
            TokenTree::Ident(self::Ident { sym, is_raw, span }) => {
                return tokenstream::TokenTree::token(Ident(sym, is_raw), span).into();
            }
            TokenTree::Literal(self::Literal {
                lit: token::Lit { kind: token::Integer, symbol, suffix },
                span,
            }) if symbol.as_str().starts_with("-") => {
                let minus = BinOp(BinOpToken::Minus);
                let symbol = Symbol::intern(&symbol.as_str()[1..]);
                let integer = TokenKind::lit(token::Integer, symbol, suffix);
                let a = tokenstream::TokenTree::token(minus, span);
                let b = tokenstream::TokenTree::token(integer, span);
                return vec![a, b].into_iter().collect();
            }
            TokenTree::Literal(self::Literal {
                lit: token::Lit { kind: token::Float, symbol, suffix },
                span,
            }) if symbol.as_str().starts_with("-") => {
                let minus = BinOp(BinOpToken::Minus);
                let symbol = Symbol::intern(&symbol.as_str()[1..]);
                let float = TokenKind::lit(token::Float, symbol, suffix);
                let a = tokenstream::TokenTree::token(minus, span);
                let b = tokenstream::TokenTree::token(float, span);
                return vec![a, b].into_iter().collect();
            }
            TokenTree::Literal(self::Literal { lit, span }) => {
                return tokenstream::TokenTree::token(Literal(lit), span).into()
            }
        };

        let kind = match ch {
            '=' => Eq,
            '<' => Lt,
            '>' => Gt,
            '!' => Not,
            '~' => Tilde,
            '+' => BinOp(Plus),
            '-' => BinOp(Minus),
            '*' => BinOp(Star),
            '/' => BinOp(Slash),
            '%' => BinOp(Percent),
            '^' => BinOp(Caret),
            '&' => BinOp(And),
            '|' => BinOp(Or),
            '@' => At,
            '.' => Dot,
            ',' => Comma,
            ';' => Semi,
            ':' => Colon,
            '#' => Pound,
            '$' => Dollar,
            '?' => Question,
            '\'' => SingleQuote,
            _ => unreachable!(),
        };

        let tree = tokenstream::TokenTree::token(kind, span);
        TokenStream::new(vec![(tree, if joint { Joint } else { NonJoint })])
    }
}

impl ToInternal<errors::Level> for Level {
    fn to_internal(self) -> errors::Level {
        match self {
            Level::Error => errors::Level::Error,
            Level::Warning => errors::Level::Warning,
            Level::Note => errors::Level::Note,
            Level::Help => errors::Level::Help,
            _ => unreachable!("unknown proc_macro::Level variant: {:?}", self),
        }
    }
}

#[derive(Clone)]
pub struct TokenStreamIter {
    cursor: tokenstream::Cursor,
    stack: Vec<TokenTree<Group, Punct, Ident, Literal>>,
}

#[derive(Clone)]
pub struct Group {
    delimiter: Delimiter,
    stream: TokenStream,
    span: DelimSpan,
}

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct Punct {
    ch: char,
    // NB. not using `Spacing` here because it doesn't implement `Hash`.
    joint: bool,
    span: Span,
}

impl Punct {
    fn new(ch: char, joint: bool, span: Span) -> Punct {
        const LEGAL_CHARS: &[char] = &['=', '<', '>', '!', '~', '+', '-', '*', '/', '%', '^',
                                       '&', '|', '@', '.', ',', ';', ':', '#', '$', '?', '\''];
        if !LEGAL_CHARS.contains(&ch) {
            panic!("unsupported character `{:?}`", ch)
        }
        Punct { ch, joint, span }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct Ident {
    sym: Symbol,
    is_raw: bool,
    span: Span,
}

impl Ident {
    fn is_valid(string: &str) -> bool {
        let mut chars = string.chars();
        if let Some(start) = chars.next() {
            (start == '_' || start.is_xid_start())
                && chars.all(|cont| cont == '_' || cont.is_xid_continue())
        } else {
            false
        }
    }
    fn new(sym: Symbol, is_raw: bool, span: Span) -> Ident {
        let string = sym.as_str();
        if !Self::is_valid(&string) {
            panic!("`{:?}` is not a valid identifier", string)
        }
        // Get rid of gensyms to conservatively check rawness on the string contents only.
        if is_raw && !sym.as_interned_str().as_symbol().can_be_raw() {
            panic!("`{}` cannot be a raw identifier", string);
        }
        Ident { sym, is_raw, span }
    }
    fn dollar_crate(span: Span) -> Ident {
        // `$crate` is accepted as an ident only if it comes from the compiler.
        Ident { sym: kw::DollarCrate, is_raw: false, span }
    }
}

// FIXME(eddyb) `Literal` should not expose internal `Debug` impls.
#[derive(Clone, Debug)]
pub struct Literal {
    lit: token::Lit,
    span: Span,
}

pub(crate) struct Rustc<'a> {
    sess: &'a ParseSess,
    def_site: Span,
    call_site: Span,
}

impl<'a> Rustc<'a> {
    pub fn new(cx: &'a ExtCtxt<'_>) -> Self {
        // No way to determine def location for a proc macro right now, so use call location.
        let location = cx.current_expansion.mark.expn_info().unwrap().call_site;
        let to_span = |transparency| {
            location.with_ctxt(
                SyntaxContext::empty()
                    .apply_mark_with_transparency(cx.current_expansion.mark, transparency),
            )
        };
        Rustc {
            sess: cx.parse_sess,
            def_site: to_span(Transparency::Opaque),
            call_site: to_span(Transparency::Transparent),
        }
    }

    fn lit(&mut self, kind: token::LitKind, symbol: Symbol, suffix: Option<Symbol>) -> Literal {
        Literal {
            lit: token::Lit::new(kind, symbol, suffix),
            span: server::Span::call_site(self),
        }
    }
}

impl server::Types for Rustc<'_> {
    type TokenStream = TokenStream;
    type TokenStreamBuilder = tokenstream::TokenStreamBuilder;
    type TokenStreamIter = TokenStreamIter;
    type Group = Group;
    type Punct = Punct;
    type Ident = Ident;
    type Literal = Literal;
    type SourceFile = Lrc<SourceFile>;
    type MultiSpan = Vec<Span>;
    type Diagnostic = Diagnostic;
    type Span = Span;
}

impl server::TokenStream for Rustc<'_> {
    fn new(&mut self) -> Self::TokenStream {
        TokenStream::empty()
    }
    fn is_empty(&mut self, stream: &Self::TokenStream) -> bool {
        stream.is_empty()
    }
    fn from_str(&mut self, src: &str) -> Self::TokenStream {
        parse::parse_stream_from_source_str(
            FileName::proc_macro_source_code(src),
            src.to_string(),
            self.sess,
            Some(self.call_site),
        )
    }
    fn to_string(&mut self, stream: &Self::TokenStream) -> String {
        stream.to_string()
    }
    fn from_token_tree(
        &mut self,
        tree: TokenTree<Self::Group, Self::Punct, Self::Ident, Self::Literal>,
    ) -> Self::TokenStream {
        tree.to_internal()
    }
    fn into_iter(&mut self, stream: Self::TokenStream) -> Self::TokenStreamIter {
        TokenStreamIter {
            cursor: stream.trees(),
            stack: vec![],
        }
    }
}

impl server::TokenStreamBuilder for Rustc<'_> {
    fn new(&mut self) -> Self::TokenStreamBuilder {
        tokenstream::TokenStreamBuilder::new()
    }
    fn push(&mut self, builder: &mut Self::TokenStreamBuilder, stream: Self::TokenStream) {
        builder.push(stream);
    }
    fn build(&mut self, builder: Self::TokenStreamBuilder) -> Self::TokenStream {
        builder.build()
    }
}

impl server::TokenStreamIter for Rustc<'_> {
    fn next(
        &mut self,
        iter: &mut Self::TokenStreamIter,
    ) -> Option<TokenTree<Self::Group, Self::Punct, Self::Ident, Self::Literal>> {
        loop {
            let tree = iter.stack.pop().or_else(|| {
                let next = iter.cursor.next_with_joint()?;
                Some(TokenTree::from_internal((next, self.sess, &mut iter.stack)))
            })?;
            // HACK: The condition "dummy span + group with empty delimiter" represents an AST
            // fragment approximately converted into a token stream. This may happen, for
            // example, with inputs to proc macro attributes, including derives. Such "groups"
            // need to flattened during iteration over stream's token trees.
            // Eventually this needs to be removed in favor of keeping original token trees
            // and not doing the roundtrip through AST.
            if let TokenTree::Group(ref group) = tree {
                if group.delimiter == Delimiter::None && group.span.entire().is_dummy() {
                    iter.cursor.append(group.stream.clone());
                    continue;
                }
            }
            return Some(tree);
        }
    }
}

impl server::Group for Rustc<'_> {
    fn new(&mut self, delimiter: Delimiter, stream: Self::TokenStream) -> Self::Group {
        Group {
            delimiter,
            stream,
            span: DelimSpan::from_single(server::Span::call_site(self)),
        }
    }
    fn delimiter(&mut self, group: &Self::Group) -> Delimiter {
        group.delimiter
    }
    fn stream(&mut self, group: &Self::Group) -> Self::TokenStream {
        group.stream.clone()
    }
    fn span(&mut self, group: &Self::Group) -> Self::Span {
        group.span.entire()
    }
    fn span_open(&mut self, group: &Self::Group) -> Self::Span {
        group.span.open
    }
    fn span_close(&mut self, group: &Self::Group) -> Self::Span {
        group.span.close
    }
    fn set_span(&mut self, group: &mut Self::Group, span: Self::Span) {
        group.span = DelimSpan::from_single(span);
    }
}

impl server::Punct for Rustc<'_> {
    fn new(&mut self, ch: char, spacing: Spacing) -> Self::Punct {
        Punct::new(ch, spacing == Spacing::Joint, server::Span::call_site(self))
    }
    fn as_char(&mut self, punct: Self::Punct) -> char {
        punct.ch
    }
    fn spacing(&mut self, punct: Self::Punct) -> Spacing {
        if punct.joint {
            Spacing::Joint
        } else {
            Spacing::Alone
        }
    }
    fn span(&mut self, punct: Self::Punct) -> Self::Span {
        punct.span
    }
    fn with_span(&mut self, punct: Self::Punct, span: Self::Span) -> Self::Punct {
        Punct { span, ..punct }
    }
}

impl server::Ident for Rustc<'_> {
    fn new(&mut self, string: &str, span: Self::Span, is_raw: bool) -> Self::Ident {
        Ident::new(Symbol::intern(string), is_raw, span)
    }
    fn span(&mut self, ident: Self::Ident) -> Self::Span {
        ident.span
    }
    fn with_span(&mut self, ident: Self::Ident, span: Self::Span) -> Self::Ident {
        Ident { span, ..ident }
    }
}

impl server::Literal for Rustc<'_> {
    // FIXME(eddyb) `Literal` should not expose internal `Debug` impls.
    fn debug(&mut self, literal: &Self::Literal) -> String {
        format!("{:?}", literal)
    }
    fn integer(&mut self, n: &str) -> Self::Literal {
        self.lit(token::Integer, Symbol::intern(n), None)
    }
    fn typed_integer(&mut self, n: &str, kind: &str) -> Self::Literal {
        self.lit(token::Integer, Symbol::intern(n), Some(Symbol::intern(kind)))
    }
    fn float(&mut self, n: &str) -> Self::Literal {
        self.lit(token::Float, Symbol::intern(n), None)
    }
    fn f32(&mut self, n: &str) -> Self::Literal {
        self.lit(token::Float, Symbol::intern(n), Some(Symbol::intern("f32")))
    }
    fn f64(&mut self, n: &str) -> Self::Literal {
        self.lit(token::Float, Symbol::intern(n), Some(Symbol::intern("f64")))
    }
    fn string(&mut self, string: &str) -> Self::Literal {
        let mut escaped = String::new();
        for ch in string.chars() {
            escaped.extend(ch.escape_debug());
        }
        self.lit(token::Str, Symbol::intern(&escaped), None)
    }
    fn character(&mut self, ch: char) -> Self::Literal {
        let mut escaped = String::new();
        escaped.extend(ch.escape_unicode());
        self.lit(token::Char, Symbol::intern(&escaped), None)
    }
    fn byte_string(&mut self, bytes: &[u8]) -> Self::Literal {
        let string = bytes
            .iter()
            .cloned()
            .flat_map(ascii::escape_default)
            .map(Into::<char>::into)
            .collect::<String>();
        self.lit(token::ByteStr, Symbol::intern(&string), None)
    }
    fn span(&mut self, literal: &Self::Literal) -> Self::Span {
        literal.span
    }
    fn set_span(&mut self, literal: &mut Self::Literal, span: Self::Span) {
        literal.span = span;
    }
    fn subspan(
        &mut self,
        literal: &Self::Literal,
        start: Bound<usize>,
        end: Bound<usize>,
    ) -> Option<Self::Span> {
        let span = literal.span;
        let length = span.hi().to_usize() - span.lo().to_usize();

        let start = match start {
            Bound::Included(lo) => lo,
            Bound::Excluded(lo) => lo + 1,
            Bound::Unbounded => 0,
        };

        let end = match end {
            Bound::Included(hi) => hi + 1,
            Bound::Excluded(hi) => hi,
            Bound::Unbounded => length,
        };

        // Bounds check the values, preventing addition overflow and OOB spans.
        if start > u32::max_value() as usize
            || end > u32::max_value() as usize
            || (u32::max_value() - start as u32) < span.lo().to_u32()
            || (u32::max_value() - end as u32) < span.lo().to_u32()
            || start >= end
            || end > length
        {
            return None;
        }

        let new_lo = span.lo() + BytePos::from_usize(start);
        let new_hi = span.lo() + BytePos::from_usize(end);
        Some(span.with_lo(new_lo).with_hi(new_hi))
    }
}

impl server::SourceFile for Rustc<'_> {
    fn eq(&mut self, file1: &Self::SourceFile, file2: &Self::SourceFile) -> bool {
        Lrc::ptr_eq(file1, file2)
    }
    fn path(&mut self, file: &Self::SourceFile) -> String {
        match file.name {
            FileName::Real(ref path) => path
                .to_str()
                .expect("non-UTF8 file path in `proc_macro::SourceFile::path`")
                .to_string(),
            _ => file.name.to_string(),
        }
    }
    fn is_real(&mut self, file: &Self::SourceFile) -> bool {
        file.is_real_file()
    }
}

impl server::MultiSpan for Rustc<'_> {
    fn new(&mut self) -> Self::MultiSpan {
        vec![]
    }
    fn push(&mut self, spans: &mut Self::MultiSpan, span: Self::Span) {
        spans.push(span)
    }
}

impl server::Diagnostic for Rustc<'_> {
    fn new(&mut self, level: Level, msg: &str, spans: Self::MultiSpan) -> Self::Diagnostic {
        let mut diag = Diagnostic::new(level.to_internal(), msg);
        diag.set_span(MultiSpan::from_spans(spans));
        diag
    }
    fn sub(
        &mut self,
        diag: &mut Self::Diagnostic,
        level: Level,
        msg: &str,
        spans: Self::MultiSpan,
    ) {
        diag.sub(level.to_internal(), msg, MultiSpan::from_spans(spans), None);
    }
    fn emit(&mut self, diag: Self::Diagnostic) {
        DiagnosticBuilder::new_diagnostic(&self.sess.span_diagnostic, diag).emit()
    }
}

impl server::Span for Rustc<'_> {
    fn debug(&mut self, span: Self::Span) -> String {
        format!("{:?} bytes({}..{})", span.ctxt(), span.lo().0, span.hi().0)
    }
    fn def_site(&mut self) -> Self::Span {
        self.def_site
    }
    fn call_site(&mut self) -> Self::Span {
        self.call_site
    }
    fn source_file(&mut self, span: Self::Span) -> Self::SourceFile {
        self.sess.source_map().lookup_char_pos(span.lo()).file
    }
    fn parent(&mut self, span: Self::Span) -> Option<Self::Span> {
        span.ctxt().outer_expn_info().map(|i| i.call_site)
    }
    fn source(&mut self, span: Self::Span) -> Self::Span {
        span.source_callsite()
    }
    fn start(&mut self, span: Self::Span) -> LineColumn {
        let loc = self.sess.source_map().lookup_char_pos(span.lo());
        LineColumn {
            line: loc.line,
            column: loc.col.to_usize(),
        }
    }
    fn end(&mut self, span: Self::Span) -> LineColumn {
        let loc = self.sess.source_map().lookup_char_pos(span.hi());
        LineColumn {
            line: loc.line,
            column: loc.col.to_usize(),
        }
    }
    fn join(&mut self, first: Self::Span, second: Self::Span) -> Option<Self::Span> {
        let self_loc = self.sess.source_map().lookup_char_pos(first.lo());
        let other_loc = self.sess.source_map().lookup_char_pos(second.lo());

        if self_loc.file.name != other_loc.file.name {
            return None;
        }

        Some(first.to(second))
    }
    fn resolved_at(&mut self, span: Self::Span, at: Self::Span) -> Self::Span {
        span.with_ctxt(at.ctxt())
    }
    fn source_text(&mut self,  span: Self::Span) -> Option<String> {
        self.sess.source_map().span_to_snippet(span).ok()
    }
}
