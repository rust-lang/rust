use crate::base::ExtCtxt;
use pm::bridge::{
    server, DelimSpan, Diagnostic, ExpnGlobals, Group, Ident, LitKind, Literal, Punct, TokenTree,
};
use pm::{Delimiter, Level};
use rustc_ast as ast;
use rustc_ast::token;
use rustc_ast::tokenstream::{self, Spacing::*, TokenStream};
use rustc_ast::util::literal::escape_byte_str_symbol;
use rustc_ast_pretty::pprust;
use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::sync::Lrc;
use rustc_errors::{MultiSpan, PResult};
use rustc_parse::lexer::nfc_normalize;
use rustc_parse::parse_stream_from_source_str;
use rustc_session::parse::ParseSess;
use rustc_span::def_id::CrateNum;
use rustc_span::symbol::{self, sym, Symbol};
use rustc_span::{BytePos, FileName, Pos, SourceFile, Span};
use smallvec::{smallvec, SmallVec};
use std::ops::{Bound, Range};

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
            token::CStr => LitKind::CStr,
            token::CStrRaw(n) => LitKind::CStrRaw(n),
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
            LitKind::CStr => token::CStr,
            LitKind::CStrRaw(n) => token::CStrRaw(n),
            LitKind::Err => token::Err,
        }
    }
}

impl FromInternal<(TokenStream2, &mut Rustc<'_, '_>)>
    for Vec<TokenTree<TokenStream2, Span, Symbol>>
{
    fn from_internal((stream, rustc): (TokenStream2, &mut Rustc<'_, '_>)) -> Self {
        use rustc_ast::token::*;

        // Estimate the capacity as `stream.len()` rounded up to the next power
        // of two to limit the number of required reallocations.
        let mut trees = Vec::with_capacity(stream.len().next_power_of_two());
        let mut cursor = stream.0.iter();

        while let Some(tree) = cursor.next() {
            let (Token { kind, span }, joint) = match tree.clone() {
                TokenTree2::Delimited(span, delim, tts) => {
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
                TokenTree2::Token(token, spacing) => (token, spacing == Joint),
            };

            // Split the operator into one or more `Punct`s, one per character.
            // The final one inherits the jointness of the original token. Any
            // before that get `joint = true`.
            let mut op = |s: &str| {
                assert!(s.is_ascii());
                trees.extend(s.bytes().enumerate().map(|(i, ch)| {
                    let is_final = i == s.len() - 1;
                    // Split the token span into single chars. Unless the span
                    // is an unusual one, e.g. due to proc macro expansion. We
                    // determine this by assuming any span with a length that
                    // matches the operator length is a normal one, and any
                    // span with a different length is an unusual one.
                    let span = if (span.hi() - span.lo()).to_usize() == s.len() {
                        let lo = span.lo() + BytePos::from_usize(i);
                        let hi = lo + BytePos::from_usize(1);
                        span.with_lo(lo).with_hi(hi)
                    } else {
                        span
                    };
                    TokenTree::Punct(Punct { ch, joint: if is_final { joint } else { true }, span })
                }));
            };

            match kind {
                Eq => op("="),
                Lt => op("<"),
                Le => op("<="),
                EqEq => op("=="),
                Ne => op("!="),
                Ge => op(">="),
                Gt => op(">"),
                AndAnd => op("&&"),
                OrOr => op("||"),
                Not => op("!"),
                Tilde => op("~"),
                BinOp(Plus) => op("+"),
                BinOp(Minus) => op("-"),
                BinOp(Star) => op("*"),
                BinOp(Slash) => op("/"),
                BinOp(Percent) => op("%"),
                BinOp(Caret) => op("^"),
                BinOp(And) => op("&"),
                BinOp(Or) => op("|"),
                BinOp(Shl) => op("<<"),
                BinOp(Shr) => op(">>"),
                BinOpEq(Plus) => op("+="),
                BinOpEq(Minus) => op("-="),
                BinOpEq(Star) => op("*="),
                BinOpEq(Slash) => op("/="),
                BinOpEq(Percent) => op("%="),
                BinOpEq(Caret) => op("^="),
                BinOpEq(And) => op("&="),
                BinOpEq(Or) => op("|="),
                BinOpEq(Shl) => op("<<="),
                BinOpEq(Shr) => op(">>="),
                At => op("@"),
                Dot => op("."),
                DotDot => op(".."),
                DotDotDot => op("..."),
                DotDotEq => op("..="),
                Comma => op(","),
                Semi => op(";"),
                Colon => op(":"),
                ModSep => op("::"),
                RArrow => op("->"),
                LArrow => op("<-"),
                FatArrow => op("=>"),
                Pound => op("#"),
                Dollar => op("$"),
                Question => op("?"),
                SingleQuote => op("'"),

                Ident(sym, is_raw) => trees.push(TokenTree::Ident(Ident { sym, is_raw, span })),
                Lifetime(name) => {
                    let ident = symbol::Ident::new(name, span).without_first_quote();
                    trees.extend([
                        TokenTree::Punct(Punct { ch: b'\'', joint: true, span }),
                        TokenTree::Ident(Ident { sym: ident.name, is_raw: false, span }),
                    ]);
                }
                Literal(token::Lit { kind, symbol, suffix }) => {
                    trees.push(TokenTree::Literal(self::Literal {
                        kind: FromInternal::from_internal(kind),
                        symbol,
                        suffix,
                        span,
                    }));
                }
                DocComment(_, attr_style, data) => {
                    let mut escaped = String::new();
                    for ch in data.as_str().chars() {
                        escaped.extend(ch.escape_debug());
                    }
                    let stream = [
                        Ident(sym::doc, false),
                        Eq,
                        TokenKind::lit(token::Str, Symbol::intern(&escaped), None),
                    ]
                    .into_iter()
                    .map(|kind| TokenTree2::token_alone(kind, span))
                    .collect();
                    trees.push(TokenTree::Punct(Punct { ch: b'#', joint: false, span }));
                    if attr_style == ast::AttrStyle::Inner {
                        trees.push(TokenTree::Punct(Punct { ch: b'!', joint: false, span }));
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
                    let stream = TokenStream2::from_internal(TokenStream::from_nonterminal_ast(&nt), true);
                    // A hack used to pass AST fragments to attribute and derive
                    // macros as a single nonterminal token instead of a token
                    // stream. Such token needs to be "unwrapped" and not
                    // represented as a delimited group.
                    // FIXME: It needs to be removed, but there are some
                    // compatibility issues (see #73345).
                    if crate::base::nt_pretty_printing_compatibility_hack(&nt, rustc.sess()) {
                        trees.extend(Self::from_internal((stream, rustc)));
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

// We use a `SmallVec` because the output size is always one or two `TokenTree`s.
impl ToInternal<SmallVec<[TokenTree2; 2]>>
    for (TokenTree<TokenStream2, Span, Symbol>, &mut Rustc<'_, '_>)
{
    fn to_internal(self) -> SmallVec<[TokenTree2; 2]> {
        use rustc_ast::token::*;

        let (tree, rustc) = self;
        match tree {
            TokenTree::Punct(Punct { ch, joint, span }) => {
                let kind = match ch {
                    b'=' => Eq,
                    b'<' => Lt,
                    b'>' => Gt,
                    b'!' => Not,
                    b'~' => Tilde,
                    b'+' => BinOp(Plus),
                    b'-' => BinOp(Minus),
                    b'*' => BinOp(Star),
                    b'/' => BinOp(Slash),
                    b'%' => BinOp(Percent),
                    b'^' => BinOp(Caret),
                    b'&' => BinOp(And),
                    b'|' => BinOp(Or),
                    b'@' => At,
                    b'.' => Dot,
                    b',' => Comma,
                    b';' => Semi,
                    b':' => Colon,
                    b'#' => Pound,
                    b'$' => Dollar,
                    b'?' => Question,
                    b'\'' => SingleQuote,
                    _ => unreachable!(),
                };
                smallvec![if joint {
                    TokenTree2::token_joint(kind, span)
                } else {
                    TokenTree2::token_alone(kind, span)
                }]
            }
            TokenTree::Group(Group { delimiter, stream, span: DelimSpan { open, close, .. } }) => {
                smallvec![TokenTree2::Delimited(
                    tokenstream::DelimSpan { open, close },
                    delimiter.to_internal(),
                    stream.unwrap_or_default(),
                )]
            }
            TokenTree::Ident(self::Ident { sym, is_raw, span }) => {
                rustc.sess().symbol_gallery.insert(sym, span);
                smallvec![TokenTree2::token_alone(Ident(sym, is_raw), span)]
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
                let a = TokenTree2::token_alone(minus, span);
                let b = TokenTree2::token_alone(integer, span);
                smallvec![a, b]
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
                let a = TokenTree2::token_alone(minus, span);
                let b = TokenTree2::token_alone(float, span);
                smallvec![a, b]
            }
            TokenTree::Literal(self::Literal { kind, symbol, suffix, span }) => {
                smallvec![TokenTree2::token_alone(
                    TokenKind::lit(kind.to_internal(), symbol, suffix),
                    span,
                )]
            }
        }
    }
}

impl ToInternal<rustc_errors::Level> for Level {
    fn to_internal(self) -> rustc_errors::Level {
        match self {
            Level::Error => rustc_errors::Level::Error { lint: false },
            Level::Warning => rustc_errors::Level::Warning(None),
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

#[derive(Clone, Debug, Default, Encodable, Decodable)]
pub struct TokenStream2(Lrc<Vec<TokenTree2>>);

#[derive(Debug, Clone, Encodable, Decodable)]
pub enum TokenTree2 {
    /// A single token. Should never be `OpenDelim` or `CloseDelim`, because
    /// delimiters are implicitly represented by `Delimited`.
    Token(rustc_ast::token::Token, rustc_ast::tokenstream::Spacing),
    /// A delimited sequence of token trees.
    Delimited(rustc_ast::tokenstream::DelimSpan, rustc_ast::token::Delimiter, TokenStream2),
}

impl FromIterator<TokenTree2> for TokenStream2 {
    fn from_iter<I: IntoIterator<Item = TokenTree2>>(iter: I) -> Self {
        TokenStream2::new(iter.into_iter().collect::<Vec<TokenTree2>>(), false)
    }
}

impl TokenStream2 {
    pub fn new(mut streams: Vec<TokenTree2>, censored: bool) -> TokenStream2 {
        if censored {
            for i in 0..streams.len() {
                if !matches!(streams.get(i + 1), Some(TokenTree2::Token(t, _)) if t.is_op()) {
                    if let TokenTree2::Token(_, spacing) = &mut streams[i] {
                        *spacing = tokenstream::Spacing::Alone;
                    }
                }
            }
        }
        TokenStream2(Lrc::new(streams))
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn from_internal(mut stream: TokenStream, censored: bool) -> Self {
        TokenStream2::new(
            std::mem::take(Lrc::make_mut(&mut stream.0))
                .into_iter()
                .map(|tree| TokenTree2::from_internal(tree, censored))
                .collect(),
            censored,
        )
    }

    pub fn to_internal(mut self) -> TokenStream {
        TokenStream::new(
            std::mem::take(Lrc::make_mut(&mut self.0))
                .into_iter()
                .map(|tree| tree.to_internal())
                .collect(),
        )
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn token_alone(kind: token::TokenKind, span: Span) -> TokenStream2 {
        TokenStream2::new(vec![TokenTree2::token_alone(kind, span)], true)
    }

    fn try_glue_to_last(vec: &mut Vec<TokenTree2>, tt: &TokenTree2) -> bool {
        if let Some(TokenTree2::Token(last_tok, tokenstream::Spacing::Joint)) = vec.last()
            && let TokenTree2::Token(tok, spacing) = tt
            && let Some(glued_tok) = last_tok.glue(tok)
        {
            // ...then overwrite the last token tree in `vec` with the
            // glued token, and skip the first token tree from `stream`.
            *vec.last_mut().unwrap() = TokenTree2::Token(glued_tok, *spacing);
            true
        } else {
            false
        }
    }

    /// Push `tt` onto the end of the stream, possibly gluing it to the last
    /// token. Uses `make_mut` to maximize efficiency.
    pub fn push_tree(&mut self, tt: TokenTree2) {
        let vec_mut = Lrc::make_mut(&mut self.0);

        if Self::try_glue_to_last(vec_mut, &tt) {
            // nothing else to do
        } else {
            vec_mut.push(tt);
        }
    }

    /// Push `stream` onto the end of the stream, possibly gluing the first
    /// token tree to the last token. (No other token trees will be glued.)
    /// Uses `make_mut` to maximize efficiency.
    pub fn push_stream(&mut self, stream: TokenStream2) {
        let vec_mut = Lrc::make_mut(&mut self.0);

        let stream_iter = stream.0.iter().cloned();

        if let Some(first) = stream.0.first() && Self::try_glue_to_last(vec_mut, first) {
            // Now skip the first token tree from `stream`.
            vec_mut.extend(stream_iter.skip(1));
        } else {
            // Append all of `stream`.
            vec_mut.extend(stream_iter);
        }
    }
}

impl TokenTree2 {
    pub fn from_internal(tree: tokenstream::TokenTree, censored: bool) -> Self {
        match tree {
            tokenstream::TokenTree::Token(token, spacing) => TokenTree2::Token(token, spacing),
            tokenstream::TokenTree::Delimited(dspan, delim, stream) => {
                TokenTree2::Delimited(dspan, delim, TokenStream2::from_internal(stream, censored))
            }
        }
    }

    pub fn to_internal(self) -> tokenstream::TokenTree {
        match self {
            TokenTree2::Token(token, spacing) => tokenstream::TokenTree::Token(token, spacing),
            TokenTree2::Delimited(dspan, delim, stream) => {
                tokenstream::TokenTree::Delimited(dspan, delim, stream.to_internal())
            }
        }
    }

    pub fn token_alone(kind: token::TokenKind, span: Span) -> TokenTree2 {
        TokenTree2::Token(
            rustc_ast::token::Token::new(kind, span),
            rustc_ast::tokenstream::Spacing::Alone,
        )
    }

    pub fn token_joint(kind: token::TokenKind, span: Span) -> TokenTree2 {
        TokenTree2::Token(
            rustc_ast::token::Token::new(kind, span),
            rustc_ast::tokenstream::Spacing::Joint,
        )
    }
}

impl server::Types for Rustc<'_, '_> {
    type FreeFunctions = FreeFunctions;
    type TokenStream = TokenStream2;
    type SourceFile = Lrc<SourceFile>;
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
                | token::LitKind::CStr
                | token::LitKind::CStrRaw(_)
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

    fn emit_diagnostic(&mut self, diagnostic: Diagnostic<Self::Span>) {
        let mut diag =
            rustc_errors::Diagnostic::new(diagnostic.level.to_internal(), diagnostic.message);
        diag.set_span(MultiSpan::from_spans(diagnostic.spans));
        for child in diagnostic.children {
            diag.sub(
                child.level.to_internal(),
                child.message,
                MultiSpan::from_spans(child.spans),
                None,
            );
        }
        self.sess().span_diagnostic.emit_diagnostic(&mut diag);
    }
}

impl server::TokenStream for Rustc<'_, '_> {
    fn is_empty(&mut self, stream: &Self::TokenStream) -> bool {
        stream.is_empty()
    }

    fn from_str(&mut self, src: &str) -> Self::TokenStream {
        TokenStream2::from_internal(
            parse_stream_from_source_str(
                FileName::proc_macro_source_code(src),
                src.to_string(),
                self.sess(),
                Some(self.call_site),
            ),
            true,
        )
    }

    fn to_string(&mut self, stream: &Self::TokenStream) -> String {
        pprust::tts_to_string(&stream.clone().to_internal())
    }

    fn expand_expr(&mut self, stream: &Self::TokenStream) -> Result<Self::TokenStream, ()> {
        // Parse the expression from our tokenstream.
        let expr: PResult<'_, _> = try {
            let mut p = rustc_parse::stream_to_parser(
                self.sess(),
                stream.clone().to_internal(),
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
            ast::ExprKind::Lit(token_lit) if token_lit.kind == token::Bool => {
                Ok(TokenStream2::token_alone(token::Ident(token_lit.symbol, false), expr.span))
            }
            ast::ExprKind::Lit(token_lit) => {
                Ok(TokenStream2::token_alone(token::Literal(*token_lit), expr.span))
            }
            ast::ExprKind::IncludedBytes(bytes) => {
                let lit = token::Lit::new(token::ByteStr, escape_byte_str_symbol(bytes), None);
                Ok(TokenStream2::token_alone(token::TokenKind::Literal(lit), expr.span))
            }
            ast::ExprKind::Unary(ast::UnOp::Neg, e) => match &e.kind {
                ast::ExprKind::Lit(token_lit) => match token_lit {
                    token::Lit { kind: token::Integer | token::Float, .. } => {
                        Ok(Self::TokenStream::from_iter([
                            // FIXME: The span of the `-` token is lost when
                            // parsing, so we cannot faithfully recover it here.
                            TokenTree2::token_alone(token::BinOp(token::Minus), e.span),
                            TokenTree2::token_alone(token::Literal(*token_lit), e.span),
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
        Self::TokenStream::new(
            (tree, &mut *self).to_internal().into_iter().collect::<Vec<_>>(),
            false,
        )
    }

    fn concat_trees(
        &mut self,
        base: Option<Self::TokenStream>,
        trees: Vec<TokenTree<Self::TokenStream, Self::Span, Self::Symbol>>,
    ) -> Self::TokenStream {
        let mut stream = if let Some(base) = base { base } else { TokenStream2::default() };
        for tree in trees {
            for tt in (tree, &mut *self).to_internal() {
                stream.push_tree(tt);
            }
        }
        stream
    }

    fn concat_streams(
        &mut self,
        base: Option<Self::TokenStream>,
        streams: Vec<Self::TokenStream>,
    ) -> Self::TokenStream {
        let mut stream = if let Some(base) = base { base } else { TokenStream2::default() };
        for s in streams {
            stream.push_stream(s);
        }
        stream
    }

    fn into_trees(
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
        match &file.name {
            FileName::Real(name) => name
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

impl server::Span for Rustc<'_, '_> {
    fn debug(&mut self, span: Self::Span) -> String {
        if self.ecx.ecfg.span_debug {
            format!("{span:?}")
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

    fn byte_range(&mut self, span: Self::Span) -> Range<usize> {
        let source_map = self.sess().source_map();

        let relative_start_pos = source_map.lookup_byte_offset(span.lo()).pos;
        let relative_end_pos = source_map.lookup_byte_offset(span.hi()).pos;

        Range { start: relative_start_pos.0 as usize, end: relative_end_pos.0 as usize }
    }
    fn start(&mut self, span: Self::Span) -> Self::Span {
        span.shrink_to_lo()
    }

    fn end(&mut self, span: Self::Span) -> Self::Span {
        span.shrink_to_hi()
    }

    fn line(&mut self, span: Self::Span) -> usize {
        let loc = self.sess().source_map().lookup_char_pos(span.lo());
        loc.line
    }

    fn column(&mut self, span: Self::Span) -> usize {
        let loc = self.sess().source_map().lookup_char_pos(span.lo());
        loc.col.to_usize() + 1
    }

    fn join(&mut self, first: Self::Span, second: Self::Span) -> Option<Self::Span> {
        let self_loc = self.sess().source_map().lookup_char_pos(first.lo());
        let other_loc = self.sess().source_map().lookup_char_pos(second.lo());

        if self_loc.file.name != other_loc.file.name {
            return None;
        }

        Some(first.to(second))
    }

    fn subspan(
        &mut self,
        span: Self::Span,
        start: Bound<usize>,
        end: Bound<usize>,
    ) -> Option<Self::Span> {
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

impl server::Symbol for Rustc<'_, '_> {
    fn normalize_and_validate_ident(&mut self, string: &str) -> Result<Self::Symbol, ()> {
        let sym = nfc_normalize(string);
        if rustc_lexer::is_ident(sym.as_str()) { Ok(sym) } else { Err(()) }
    }
}

impl server::Server for Rustc<'_, '_> {
    fn globals(&mut self) -> ExpnGlobals<Self::Span> {
        ExpnGlobals {
            def_site: self.def_site,
            call_site: self.call_site,
            mixed_site: self.mixed_site,
        }
    }

    fn intern_symbol(string: &str) -> Self::Symbol {
        Symbol::intern(string)
    }

    fn with_symbol_string(symbol: &Self::Symbol, f: impl FnOnce(&str)) {
        f(&symbol.as_str())
    }
}
