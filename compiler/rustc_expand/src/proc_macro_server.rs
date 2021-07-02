use crate::base::ExtCtxt;

use rustc_ast as ast;
use rustc_ast::token;
use rustc_ast::tokenstream::{self, Spacing::*, TokenStream};
use rustc_ast_pretty::pprust;
use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::sync::Lrc;
use rustc_errors::{Diagnostic, MultiSpan, PResult};
use rustc_parse::parse_stream_from_source_str;
use rustc_session::parse::ParseSess;
use rustc_span::def_id::CrateNum;
use rustc_span::symbol::{self, sym, Symbol};
use rustc_span::{BytePos, FileName, Pos, SourceFile, Span};

use pm::bridge::{server, DelimSpan, Group, Ident, LitKind, Literal, Punct, TokenTree};
use pm::{Delimiter, Level, LineColumn};
use std::ops::Bound;

trait FromInternal<T> {
    fn from_internal(x: T) -> Self;
}

trait ToInternal<T> {
    fn to_internal(self) -> T;
}

impl FromInternal<token::Delimiter> for Delimiter {
    fn from_internal(delim: token::Delimiter) -> Delimiter {
        match delim {
            token::Delimiter::Parenthesis => Delimiter::Parenthesis,
            token::Delimiter::Brace => Delimiter::Brace,
            token::Delimiter::Bracket => Delimiter::Bracket,
            token::Delimiter::Invisible => Delimiter::None,
        }
    }
}

impl ToInternal<token::Delimiter> for Delimiter {
    fn to_internal(self) -> token::Delimiter {
        match self {
            Delimiter::Parenthesis => token::Delimiter::Parenthesis,
            Delimiter::Brace => token::Delimiter::Brace,
            Delimiter::Bracket => token::Delimiter::Bracket,
            Delimiter::None => token::Delimiter::Invisible,
        }
    }
}

impl FromInternal<token::LitKind> for LitKind {
    fn from_internal(kind: token::LitKind) -> Self {
        match kind {
            token::Byte => LitKind::Byte,
            token::Char => LitKind::Char,
            token::Integer => LitKind::Integer,
            token::Float => LitKind::Float,
            token::Str => LitKind::Str,
            token::StrRaw(n) => LitKind::StrRaw(n),
            token::ByteStr => LitKind::ByteStr,
            token::ByteStrRaw(n) => LitKind::ByteStrRaw(n),
            token::Err => LitKind::Err,
            token::Bool => unreachable!(),
        }
    }
}

impl ToInternal<token::LitKind> for LitKind {
    fn to_internal(self) -> token::LitKind {
        match self {
            LitKind::Byte => token::Byte,
            LitKind::Char => token::Char,
            LitKind::Integer => token::Integer,
            LitKind::Float => token::Float,
            LitKind::Str => token::Str,
            LitKind::StrRaw(n) => token::StrRaw(n),
            LitKind::ByteStr => token::ByteStr,
            LitKind::ByteStrRaw(n) => token::ByteStrRaw(n),
            LitKind::Err => token::Err,
        }
    }
}

impl FromInternal<(TokenStream, &mut Rustc<'_, '_>)> for Vec<TokenTree<TokenStream, Span, Symbol>> {
    fn from_internal((stream, rustc): (TokenStream, &mut Rustc<'_, '_>)) -> Self {
        use rustc_ast::token::*;

        let mut cursor = stream.into_trees();
        let mut trees = Vec::new();

        while let Some((tree, spacing)) = cursor.next_with_spacing() {
            let joint = spacing == Joint;
            let Token { kind, span } = match tree {
                tokenstream::TokenTree::Delimited(span, delim, tts) => {
                    let delimiter = pm::Delimiter::from_internal(delim);
                    trees.push(TokenTree::Group(Group {
                        delimiter,
                        stream: Some(tts),
                        span: DelimSpan {
                            open: span.open,
                            close: span.close,
                            entire: span.entire(),
                        },
                    }));
                    continue;
                }
                tokenstream::TokenTree::Token(token) => token,
            };

            macro_rules! tt {
                ($ty:ident { $($field:ident $(: $value:expr)*),+ $(,)? }) => (
                    trees.push(TokenTree::$ty(self::$ty {
                        $($field $(: $value)*,)+
                        span,
                    }))
                );
            }
            macro_rules! op {
                ($a:expr) => {{
                    tt!(Punct { ch: $a, joint });
                }};
                ($a:expr, $b:expr) => {{
                    tt!(Punct { ch: $a, joint: true });
                    tt!(Punct { ch: $b, joint });
                }};
                ($a:expr, $b:expr, $c:expr) => {{
                    tt!(Punct { ch: $a, joint: true });
                    tt!(Punct { ch: $b, joint: true });
                    tt!(Punct { ch: $c, joint });
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

                Ident(sym, is_raw) => tt!(Ident { sym, is_raw }),
                Lifetime(name) => {
                    let ident = symbol::Ident::new(name, span).without_first_quote();
                    tt!(Punct { ch: '\'', joint: true });
                    tt!(Ident { sym: ident.name, is_raw: false });
                }
                Literal(token::Lit { kind, symbol, suffix }) => {
                    tt!(Literal { kind: FromInternal::from_internal(kind), symbol, suffix });
                }
                DocComment(_, attr_style, data) => {
                    let mut escaped = String::new();
                    for ch in data.as_str().chars() {
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
                    tt!(Punct { ch: '#', joint: false });
                    if attr_style == ast::AttrStyle::Inner {
                        tt!(Punct { ch: '!', joint: false });
                    }
                    trees.push(TokenTree::Group(Group {
                        delimiter: pm::Delimiter::Bracket,
                        stream: Some(stream),
                        span: DelimSpan::from_single(span),
                    }));
                }

                Interpolated(nt) if let NtIdent(ident, is_raw) = *nt => {
                    trees.push(TokenTree::Ident(Ident { sym: ident.name, is_raw, span: ident.span }))
                }

                Interpolated(nt) => {
                    let stream = TokenStream::from_nonterminal_ast(&nt);
                    if crate::base::nt_pretty_printing_compatibility_hack(&nt, rustc.sess()) {
                        cursor.append(stream);
                    } else {
                        trees.push(TokenTree::Group(Group {
                            delimiter: pm::Delimiter::None,
                            stream: Some(stream),
                            span: DelimSpan::from_single(span),
                        }))
                    }
                }

                OpenDelim(..) | CloseDelim(..) => unreachable!(),
                Eof => unreachable!(),
            }
        }
        trees
    }
}

impl ToInternal<TokenStream> for TokenTree<TokenStream, Span, Symbol> {
    fn to_internal(self) -> TokenStream {
        use rustc_ast::token::*;

        let (ch, joint, span) = match self {
            TokenTree::Punct(Punct { ch, joint, span }) => (ch, joint, span),
            TokenTree::Group(Group { delimiter, stream, span: DelimSpan { open, close, .. } }) => {
                return tokenstream::TokenTree::Delimited(
                    tokenstream::DelimSpan { open, close },
                    delimiter.to_internal(),
                    stream.unwrap_or_default(),
                )
                .into();
            }
            TokenTree::Ident(self::Ident { sym, is_raw, span }) => {
                return tokenstream::TokenTree::token(Ident(sym, is_raw), span).into();
            }
            TokenTree::Literal(self::Literal {
                kind: self::LitKind::Integer,
                symbol,
                suffix,
                span,
            }) if symbol.as_str().starts_with('-') => {
                let minus = BinOp(BinOpToken::Minus);
                let symbol = Symbol::intern(&symbol.as_str()[1..]);
                let integer = TokenKind::lit(token::Integer, symbol, suffix);
                let a = tokenstream::TokenTree::token(minus, span);
                let b = tokenstream::TokenTree::token(integer, span);
                return [a, b].into_iter().collect();
            }
            TokenTree::Literal(self::Literal {
                kind: self::LitKind::Float,
                symbol,
                suffix,
                span,
            }) if symbol.as_str().starts_with('-') => {
                let minus = BinOp(BinOpToken::Minus);
                let symbol = Symbol::intern(&symbol.as_str()[1..]);
                let float = TokenKind::lit(token::Float, symbol, suffix);
                let a = tokenstream::TokenTree::token(minus, span);
                let b = tokenstream::TokenTree::token(float, span);
                return [a, b].into_iter().collect();
            }
            TokenTree::Literal(self::Literal { kind, symbol, suffix, span }) => {
                return tokenstream::TokenTree::token(
                    TokenKind::lit(kind.to_internal(), symbol, suffix),
                    span,
                )
                .into();
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
        TokenStream::new(vec![(tree, if joint { Joint } else { Alone })])
    }
}

impl ToInternal<rustc_errors::Level> for Level {
    fn to_internal(self) -> rustc_errors::Level {
        match self {
            Level::Error => rustc_errors::Level::Error { lint: false },
            Level::Warning => rustc_errors::Level::Warning,
            Level::Note => rustc_errors::Level::Note,
            Level::Help => rustc_errors::Level::Help,
            _ => unreachable!("unknown proc_macro::Level variant: {:?}", self),
        }
    }
}

pub struct FreeFunctions;

pub(crate) struct Rustc<'a, 'b> {
    ecx: &'a mut ExtCtxt<'b>,
    def_site: Span,
    call_site: Span,
    mixed_site: Span,
    krate: CrateNum,
    rebased_spans: FxHashMap<usize, Span>,
}

impl<'a, 'b> Rustc<'a, 'b> {
    pub fn new(ecx: &'a mut ExtCtxt<'b>) -> Self {
        let expn_data = ecx.current_expansion.id.expn_data();
        Rustc {
            def_site: ecx.with_def_site_ctxt(expn_data.def_site),
            call_site: ecx.with_call_site_ctxt(expn_data.call_site),
            mixed_site: ecx.with_mixed_site_ctxt(expn_data.call_site),
            krate: expn_data.macro_def_id.unwrap().krate,
            rebased_spans: FxHashMap::default(),
            ecx,
        }
    }

    fn sess(&self) -> &ParseSess {
        self.ecx.parse_sess()
    }
}

impl server::Types for Rustc<'_, '_> {
    type FreeFunctions = FreeFunctions;
    type TokenStream = TokenStream;
    type SourceFile = Lrc<SourceFile>;
    type MultiSpan = Vec<Span>;
    type Diagnostic = Diagnostic;
    type Span = Span;
    type Symbol = Symbol;
}

impl server::FreeFunctions for Rustc<'_, '_> {
    fn track_env_var(&mut self, var: &str, value: Option<&str>) {
        self.sess()
            .env_depinfo
            .borrow_mut()
            .insert((Symbol::intern(var), value.map(Symbol::intern)));
    }

    fn track_path(&mut self, path: &str) {
        self.sess().file_depinfo.borrow_mut().insert(Symbol::intern(path));
    }

    fn literal_from_str(&mut self, s: &str) -> Result<Literal<Self::Span, Self::Symbol>, ()> {
        let name = FileName::proc_macro_source_code(s);
        let mut parser = rustc_parse::new_parser_from_source_str(self.sess(), name, s.to_owned());

        let first_span = parser.token.span.data();
        let minus_present = parser.eat(&token::BinOp(token::Minus));

        let lit_span = parser.token.span.data();
        let token::Literal(mut lit) = parser.token.kind else {
            return Err(());
        };

        // Check no comment or whitespace surrounding the (possibly negative)
        // literal, or more tokens after it.
        if (lit_span.hi.0 - first_span.lo.0) as usize != s.len() {
            return Err(());
        }

        if minus_present {
            // If minus is present, check no comment or whitespace in between it
            // and the literal token.
            if first_span.hi.0 != lit_span.lo.0 {
                return Err(());
            }

            // Check literal is a kind we allow to be negated in a proc macro token.
            match lit.kind {
                token::LitKind::Bool
                | token::LitKind::Byte
                | token::LitKind::Char
                | token::LitKind::Str
                | token::LitKind::StrRaw(_)
                | token::LitKind::ByteStr
                | token::LitKind::ByteStrRaw(_)
                | token::LitKind::Err => return Err(()),
                token::LitKind::Integer | token::LitKind::Float => {}
            }

            // Synthesize a new symbol that includes the minus sign.
            let symbol = Symbol::intern(&s[..1 + lit.symbol.as_str().len()]);
            lit = token::Lit::new(lit.kind, symbol, lit.suffix);
        }
        let token::Lit { kind, symbol, suffix } = lit;
        Ok(Literal {
            kind: FromInternal::from_internal(kind),
            symbol,
            suffix,
            span: self.call_site,
        })
    }

    fn literal_subspan(
        &mut self,
        literal: Literal<Self::Span, Self::Symbol>,
        start: Bound<usize>,
        end: Bound<usize>,
    ) -> Option<Self::Span> {
        let span = literal.span;
        let length = span.hi().to_usize() - span.lo().to_usize();

        let start = match start {
            Bound::Included(lo) => lo,
            Bound::Excluded(lo) => lo.checked_add(1)?,
            Bound::Unbounded => 0,
        };

        let end = match end {
            Bound::Included(hi) => hi.checked_add(1)?,
            Bound::Excluded(hi) => hi,
            Bound::Unbounded => length,
        };

        // Bounds check the values, preventing addition overflow and OOB spans.
        if start > u32::MAX as usize
            || end > u32::MAX as usize
            || (u32::MAX - start as u32) < span.lo().to_u32()
            || (u32::MAX - end as u32) < span.lo().to_u32()
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

impl server::TokenStream for Rustc<'_, '_> {
    fn is_empty(&mut self, stream: &Self::TokenStream) -> bool {
        stream.is_empty()
    }
    fn from_str(&mut self, src: &str) -> Self::TokenStream {
        parse_stream_from_source_str(
            FileName::proc_macro_source_code(src),
            src.to_string(),
            self.sess(),
            Some(self.call_site),
        )
    }
    fn to_string(&mut self, stream: &Self::TokenStream) -> String {
        pprust::tts_to_string(stream)
    }
    fn expand_expr(&mut self, stream: &Self::TokenStream) -> Result<Self::TokenStream, ()> {
        // Parse the expression from our tokenstream.
        let expr: PResult<'_, _> = try {
            let mut p = rustc_parse::stream_to_parser(
                self.sess(),
                stream.clone(),
                Some("proc_macro expand expr"),
            );
            let expr = p.parse_expr()?;
            if p.token != token::Eof {
                p.unexpected()?;
            }
            expr
        };
        let expr = expr.map_err(|mut err| {
            err.emit();
        })?;

        // Perform eager expansion on the expression.
        let expr = self
            .ecx
            .expander()
            .fully_expand_fragment(crate::expand::AstFragment::Expr(expr))
            .make_expr();

        // NOTE: For now, limit `expand_expr` to exclusively expand to literals.
        // This may be relaxed in the future.
        // We don't use `TokenStream::from_ast` as the tokenstream currently cannot
        // be recovered in the general case.
        match &expr.kind {
            ast::ExprKind::Lit(l) => {
                Ok(tokenstream::TokenTree::token(token::Literal(l.token), l.span).into())
            }
            ast::ExprKind::Unary(ast::UnOp::Neg, e) => match &e.kind {
                ast::ExprKind::Lit(l) => match l.token {
                    token::Lit { kind: token::Integer | token::Float, .. } => {
                        Ok(Self::TokenStream::from_iter([
                            // FIXME: The span of the `-` token is lost when
                            // parsing, so we cannot faithfully recover it here.
                            tokenstream::TokenTree::token(token::BinOp(token::Minus), e.span),
                            tokenstream::TokenTree::token(token::Literal(l.token), l.span),
                        ]))
                    }
                    _ => Err(()),
                },
                _ => Err(()),
            },
            _ => Err(()),
        }
    }
    fn from_token_tree(
        &mut self,
        tree: TokenTree<Self::TokenStream, Self::Span, Self::Symbol>,
    ) -> Self::TokenStream {
        tree.to_internal()
    }
    fn concat_trees(
        &mut self,
        base: Option<Self::TokenStream>,
        trees: Vec<TokenTree<Self::TokenStream, Self::Span, Self::Symbol>>,
    ) -> Self::TokenStream {
        let mut builder = tokenstream::TokenStreamBuilder::new();
        if let Some(base) = base {
            builder.push(base);
        }
        for tree in trees {
            builder.push(tree.to_internal());
        }
        builder.build()
    }
    fn concat_streams(
        &mut self,
        base: Option<Self::TokenStream>,
        streams: Vec<Self::TokenStream>,
    ) -> Self::TokenStream {
        let mut builder = tokenstream::TokenStreamBuilder::new();
        if let Some(base) = base {
            builder.push(base);
        }
        for stream in streams {
            builder.push(stream);
        }
        builder.build()
    }
    fn into_iter(
        &mut self,
        stream: Self::TokenStream,
    ) -> Vec<TokenTree<Self::TokenStream, Self::Span, Self::Symbol>> {
        FromInternal::from_internal((stream, self))
    }
}

impl server::SourceFile for Rustc<'_, '_> {
    fn eq(&mut self, file1: &Self::SourceFile, file2: &Self::SourceFile) -> bool {
        Lrc::ptr_eq(file1, file2)
    }
    fn path(&mut self, file: &Self::SourceFile) -> String {
        match file.name {
            FileName::Real(ref name) => name
                .local_path()
                .expect("attempting to get a file path in an imported file in `proc_macro::SourceFile::path`")
                .to_str()
                .expect("non-UTF8 file path in `proc_macro::SourceFile::path`")
                .to_string(),
            _ => file.name.prefer_local().to_string(),
        }
    }
    fn is_real(&mut self, file: &Self::SourceFile) -> bool {
        file.is_real_file()
    }
}

impl server::MultiSpan for Rustc<'_, '_> {
    fn new(&mut self) -> Self::MultiSpan {
        vec![]
    }
    fn push(&mut self, spans: &mut Self::MultiSpan, span: Self::Span) {
        spans.push(span)
    }
}

impl server::Diagnostic for Rustc<'_, '_> {
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
    fn emit(&mut self, mut diag: Self::Diagnostic) {
        self.sess().span_diagnostic.emit_diagnostic(&mut diag);
    }
}

impl server::Span for Rustc<'_, '_> {
    fn debug(&mut self, span: Self::Span) -> String {
        if self.ecx.ecfg.span_debug {
            format!("{:?}", span)
        } else {
            format!("{:?} bytes({}..{})", span.ctxt(), span.lo().0, span.hi().0)
        }
    }
    fn source_file(&mut self, span: Self::Span) -> Self::SourceFile {
        self.sess().source_map().lookup_char_pos(span.lo()).file
    }
    fn parent(&mut self, span: Self::Span) -> Option<Self::Span> {
        span.parent_callsite()
    }
    fn source(&mut self, span: Self::Span) -> Self::Span {
        span.source_callsite()
    }
    fn start(&mut self, span: Self::Span) -> LineColumn {
        let loc = self.sess().source_map().lookup_char_pos(span.lo());
        LineColumn { line: loc.line, column: loc.col.to_usize() }
    }
    fn end(&mut self, span: Self::Span) -> LineColumn {
        let loc = self.sess().source_map().lookup_char_pos(span.hi());
        LineColumn { line: loc.line, column: loc.col.to_usize() }
    }
    fn before(&mut self, span: Self::Span) -> Self::Span {
        span.shrink_to_lo()
    }
    fn after(&mut self, span: Self::Span) -> Self::Span {
        span.shrink_to_hi()
    }
    fn join(&mut self, first: Self::Span, second: Self::Span) -> Option<Self::Span> {
        let self_loc = self.sess().source_map().lookup_char_pos(first.lo());
        let other_loc = self.sess().source_map().lookup_char_pos(second.lo());

        if self_loc.file.name != other_loc.file.name {
            return None;
        }

        Some(first.to(second))
    }
    fn resolved_at(&mut self, span: Self::Span, at: Self::Span) -> Self::Span {
        span.with_ctxt(at.ctxt())
    }
    fn source_text(&mut self, span: Self::Span) -> Option<String> {
        self.sess().source_map().span_to_snippet(span).ok()
    }
    /// Saves the provided span into the metadata of
    /// *the crate we are currently compiling*, which must
    /// be a proc-macro crate. This id can be passed to
    /// `recover_proc_macro_span` when our current crate
    /// is *run* as a proc-macro.
    ///
    /// Let's suppose that we have two crates - `my_client`
    /// and `my_proc_macro`. The `my_proc_macro` crate
    /// contains a procedural macro `my_macro`, which
    /// is implemented as: `quote! { "hello" }`
    ///
    /// When we *compile* `my_proc_macro`, we will execute
    /// the `quote` proc-macro. This will save the span of
    /// "hello" into the metadata of `my_proc_macro`. As a result,
    /// the body of `my_proc_macro` (after expansion) will end
    /// up containing a call that looks like this:
    /// `proc_macro::Ident::new("hello", proc_macro::Span::recover_proc_macro_span(0))`
    ///
    /// where `0` is the id returned by this function.
    /// When `my_proc_macro` *executes* (during the compilation of `my_client`),
    /// the call to `recover_proc_macro_span` will load the corresponding
    /// span from the metadata of `my_proc_macro` (which we have access to,
    /// since we've loaded `my_proc_macro` from disk in order to execute it).
    /// In this way, we have obtained a span pointing into `my_proc_macro`
    fn save_span(&mut self, span: Self::Span) -> usize {
        self.sess().save_proc_macro_span(span)
    }
    fn recover_proc_macro_span(&mut self, id: usize) -> Self::Span {
        let (resolver, krate, def_site) = (&*self.ecx.resolver, self.krate, self.def_site);
        *self.rebased_spans.entry(id).or_insert_with(|| {
            // FIXME: `SyntaxContext` for spans from proc macro crates is lost during encoding,
            // replace it with a def-site context until we are encoding it properly.
            resolver.get_proc_macro_quoted_span(krate, id).with_ctxt(def_site.ctxt())
        })
    }
}

impl server::Context for Rustc<'_, '_> {
    fn def_site(&mut self) -> Self::Span {
        self.def_site
    }
    fn call_site(&mut self) -> Self::Span {
        self.call_site
    }
    fn mixed_site(&mut self) -> Self::Span {
        self.mixed_site
    }

    // NOTE: May be run on any thread, so cannot use `nfc_normalize`
    fn validate_ident(s: &str) -> Result<Option<String>, ()> {
        use unicode_normalization::{is_nfc_quick, IsNormalized, UnicodeNormalization};
        let normalized: Option<String> = match is_nfc_quick(s.chars()) {
            IsNormalized::Yes => None,
            _ => Some(s.chars().nfc().collect()),
        };
        if rustc_lexer::is_ident(normalized.as_ref().map(|s| &s[..]).unwrap_or(s)) {
            Ok(normalized)
        } else {
            Err(())
        }
    }

    fn intern_symbol(string: &str) -> Self::Symbol {
        Symbol::intern(string)
    }

    fn with_symbol_string(symbol: &Self::Symbol, f: impl FnOnce(&str)) {
        f(&symbol.as_str())
    }
}
