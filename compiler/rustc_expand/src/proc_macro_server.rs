use std::ops::{Bound, Range};
use std::rc::Rc;

use ast::token::IdentIsRaw;
use rustc_ast as ast;
use rustc_ast::token;
use rustc_ast::tokenstream::{self, DelimSpacing, Spacing, TokenStream};
use rustc_ast::util::literal::escape_byte_str_symbol;
use rustc_ast_pretty::pprust;
use rustc_data_structures::fx::FxHashMap;
use rustc_errors::{Diag, ErrorGuaranteed, MultiSpan};
use rustc_parse::lexer::{StripTokens, nfc_normalize};
use rustc_parse::parser::Parser;
use rustc_parse::{exp, new_parser_from_source_str, source_str_to_stream};
use rustc_proc_macro::bridge::{
    DelimSpan, Diagnostic, ExpnGlobals, Group, Ident, LitKind, Literal, Punct, TokenTree, server,
};
use rustc_proc_macro::{Delimiter, Level};
use rustc_session::parse::ParseSess;
use rustc_span::def_id::CrateNum;
use rustc_span::{BytePos, FileName, Pos, Span, Symbol, sym};

use crate::base::ExtCtxt;

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
            token::Delimiter::Invisible(_) => Delimiter::None,
        }
    }
}

impl ToInternal<token::Delimiter> for Delimiter {
    fn to_internal(self) -> token::Delimiter {
        match self {
            Delimiter::Parenthesis => token::Delimiter::Parenthesis,
            Delimiter::Brace => token::Delimiter::Brace,
            Delimiter::Bracket => token::Delimiter::Bracket,
            Delimiter::None => token::Delimiter::Invisible(token::InvisibleOrigin::ProcMacro),
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
            token::Err(_guar) => {
                // This is the only place a `rustc_proc_macro::bridge::LitKind::ErrWithGuar`
                // is constructed. Note that an `ErrorGuaranteed` is available,
                // as required. See the comment in `to_internal`.
                LitKind::ErrWithGuar
            }
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
            LitKind::ErrWithGuar => {
                // This is annoying but valid. `LitKind::ErrWithGuar` would
                // have an `ErrorGuaranteed` except that type isn't available
                // in that crate. So we have to fake one. And we don't want to
                // use a delayed bug because there might be lots of these,
                // which would be expensive.
                #[allow(deprecated)]
                let guar = ErrorGuaranteed::unchecked_error_guaranteed();
                token::Err(guar)
            }
        }
    }
}

impl FromInternal<(TokenStream, &mut Rustc<'_, '_>)> for Vec<TokenTree<Span, Symbol>> {
    fn from_internal((stream, rustc): (TokenStream, &mut Rustc<'_, '_>)) -> Self {
        use rustc_ast::token::*;

        // Estimate the capacity as `stream.len()` rounded up to the next power
        // of two to limit the number of required reallocations.
        let mut trees = Vec::with_capacity(stream.len().next_power_of_two());
        let mut iter = stream.iter();

        while let Some(tree) = iter.next() {
            let (Token { kind, span }, joint) = match tree.clone() {
                tokenstream::TokenTree::Delimited(span, _, mut delim, mut stream) => {
                    // In `mk_delimited` we avoid nesting invisible delimited
                    // of the same `MetaVarKind`. Here we do the same but
                    // ignore the `MetaVarKind` because it is discarded when we
                    // convert it to a `Group`.
                    while let Delimiter::Invisible(InvisibleOrigin::MetaVar(_)) = delim {
                        if stream.len() == 1
                            && let tree = stream.iter().next().unwrap()
                            && let tokenstream::TokenTree::Delimited(_, _, delim2, stream2) = tree
                            && let Delimiter::Invisible(InvisibleOrigin::MetaVar(_)) = delim2
                        {
                            delim = *delim2;
                            stream = stream2.clone();
                        } else {
                            break;
                        }
                    }

                    trees.push(TokenTree::Group(Group {
                        delimiter: rustc_proc_macro::Delimiter::from_internal(delim),
                        stream: Some(rustc.ts_rustc_to_pm(stream)),
                        span: DelimSpan {
                            open: span.open,
                            close: span.close,
                            entire: span.entire(),
                        },
                    }));
                    continue;
                }
                tokenstream::TokenTree::Token(token, spacing) => {
                    // Do not be tempted to check here that the `spacing`
                    // values are "correct" w.r.t. the token stream (e.g. that
                    // `Spacing::Joint` is actually followed by a `Punct` token
                    // tree). Because the problem in #76399 was introduced that
                    // way.
                    //
                    // This is where the `Hidden` in `JointHidden` applies,
                    // because the jointness is effectively hidden from proc
                    // macros.
                    let joint = match spacing {
                        Spacing::Alone | Spacing::JointHidden => false,
                        Spacing::Joint => true,
                    };
                    (token, joint)
                }
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
                    let joint = if is_final { joint } else { true };
                    TokenTree::Punct(Punct { ch, joint, span })
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
                Bang => op("!"),
                Tilde => op("~"),
                Plus => op("+"),
                Minus => op("-"),
                Star => op("*"),
                Slash => op("/"),
                Percent => op("%"),
                Caret => op("^"),
                And => op("&"),
                Or => op("|"),
                Shl => op("<<"),
                Shr => op(">>"),
                PlusEq => op("+="),
                MinusEq => op("-="),
                StarEq => op("*="),
                SlashEq => op("/="),
                PercentEq => op("%="),
                CaretEq => op("^="),
                AndEq => op("&="),
                OrEq => op("|="),
                ShlEq => op("<<="),
                ShrEq => op(">>="),
                At => op("@"),
                Dot => op("."),
                DotDot => op(".."),
                DotDotDot => op("..."),
                DotDotEq => op("..="),
                Comma => op(","),
                Semi => op(";"),
                Colon => op(":"),
                PathSep => op("::"),
                RArrow => op("->"),
                LArrow => op("<-"),
                FatArrow => op("=>"),
                Pound => op("#"),
                Dollar => op("$"),
                Question => op("?"),
                SingleQuote => op("'"),

                Ident(sym, is_raw) => trees.push(TokenTree::Ident(Ident {
                    sym,
                    is_raw: matches!(is_raw, IdentIsRaw::Yes),
                    span,
                })),
                NtIdent(ident, is_raw) => trees.push(TokenTree::Ident(Ident {
                    sym: ident.name,
                    is_raw: matches!(is_raw, IdentIsRaw::Yes),
                    span: ident.span,
                })),

                Lifetime(name, is_raw) => {
                    let ident = rustc_span::Ident::new(name, span).without_first_quote();
                    trees.extend([
                        TokenTree::Punct(Punct { ch: b'\'', joint: true, span }),
                        TokenTree::Ident(Ident {
                            sym: ident.name,
                            is_raw: matches!(is_raw, IdentIsRaw::Yes),
                            span,
                        }),
                    ]);
                }
                NtLifetime(ident, is_raw) => {
                    let stream =
                        TokenStream::token_alone(token::Lifetime(ident.name, is_raw), ident.span);
                    trees.push(TokenTree::Group(Group {
                        delimiter: rustc_proc_macro::Delimiter::None,
                        stream: Some(rustc.ts_rustc_to_pm(stream)),
                        span: DelimSpan::from_single(span),
                    }))
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
                        Ident(sym::doc, IdentIsRaw::No),
                        Eq,
                        TokenKind::lit(token::Str, Symbol::intern(&escaped), None),
                    ]
                    .into_iter()
                    .map(|kind| tokenstream::TokenTree::token_alone(kind, span))
                    .collect();
                    trees.push(TokenTree::Punct(Punct { ch: b'#', joint: false, span }));
                    if attr_style == ast::AttrStyle::Inner {
                        trees.push(TokenTree::Punct(Punct { ch: b'!', joint: false, span }));
                    }
                    trees.push(TokenTree::Group(Group {
                        delimiter: rustc_proc_macro::Delimiter::Bracket,
                        stream: Some(rustc.ts_rustc_to_pm(stream)),
                        span: DelimSpan::from_single(span),
                    }));
                }

                OpenParen | CloseParen | OpenBrace | CloseBrace | OpenBracket | CloseBracket
                | OpenInvisible(_) | CloseInvisible(_) | Eof => unreachable!(),
            }
        }
        trees
    }
}

impl ToInternal<rustc_errors::Level> for Level {
    fn to_internal(self) -> rustc_errors::Level {
        match self {
            Level::Error => rustc_errors::Level::Error,
            Level::Warning => rustc_errors::Level::Warning,
            Level::Note => rustc_errors::Level::Note,
            Level::Help => rustc_errors::Level::Help,
            _ => unreachable!("unknown proc_macro::Level variant: {:?}", self),
        }
    }
}

fn cancel_diags_into_string(diags: Vec<Diag<'_>>) -> String {
    let mut messages = diags.into_iter().flat_map(Diag::cancel_into_message);
    let msg = messages.next().expect("no diagnostic has a message");
    messages.for_each(|_| ()); // consume iterator to cancel the remaining diagnostics
    msg
}

type BridgeTokenStream = rustc_proc_macro::bridge::TokenStream<Span, Symbol>;

pub(crate) struct Rustc<'a, 'b> {
    ecx: &'a mut ExtCtxt<'b>,
    def_site: Span,
    call_site: Span,
    mixed_site: Span,
    krate: CrateNum,
    rebased_spans: FxHashMap<usize, Span>,
}

impl<'a, 'b> Rustc<'a, 'b> {
    pub(crate) fn new(ecx: &'a mut ExtCtxt<'b>) -> Self {
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

    fn psess(&self) -> &ParseSess {
        self.ecx.psess()
    }

    pub(crate) fn ts_pm_to_rustc(&mut self, ts: BridgeTokenStream) -> tokenstream::TokenStream {
        let mut t = tokenstream::TokenStream::new(vec![]);

        for tree in Rc::into_inner(ts.trees).expect("just decoded this `Rc`") {
            use rustc_ast::token::*;

            // The code below is conservative, using `token_alone`/`Spacing::Alone`
            // in most places. It's hard in general to do better when working at
            // the token level. When the resulting code is pretty-printed by
            // `print_tts` the `space_between` function helps avoid a lot of
            // unnecessary whitespace, so the results aren't too bad.
            match tree {
                TokenTree::Punct(Punct { ch, joint, span }) => {
                    let kind = match ch {
                        b'=' => Eq,
                        b'<' => Lt,
                        b'>' => Gt,
                        b'!' => Bang,
                        b'~' => Tilde,
                        b'+' => Plus,
                        b'-' => Minus,
                        b'*' => Star,
                        b'/' => Slash,
                        b'%' => Percent,
                        b'^' => Caret,
                        b'&' => And,
                        b'|' => Or,
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
                    // We never produce `token::Spacing::JointHidden` here, which
                    // means the pretty-printing of code produced by proc macros is
                    // ugly, with lots of whitespace between tokens. This is
                    // unavoidable because `proc_macro::Spacing` only applies to
                    // `Punct` token trees.
                    t.push_tree(if joint {
                        tokenstream::TokenTree::token_joint(kind, span)
                    } else {
                        tokenstream::TokenTree::token_alone(kind, span)
                    })
                }
                TokenTree::Group(Group {
                    delimiter,
                    stream,
                    span: DelimSpan { open, close, .. },
                }) => t.push_tree(tokenstream::TokenTree::Delimited(
                    tokenstream::DelimSpan { open, close },
                    DelimSpacing::new(Spacing::Alone, Spacing::Alone),
                    delimiter.to_internal(),
                    self.ts_pm_to_rustc(stream.unwrap_or_default()),
                )),
                TokenTree::Ident(self::Ident { sym, is_raw, span }) => {
                    self.psess().symbol_gallery.insert(sym, span);
                    t.push_tree(tokenstream::TokenTree::token_alone(
                        Ident(sym, is_raw.into()),
                        span,
                    ))
                }
                TokenTree::Literal(self::Literal {
                    kind: self::LitKind::Integer,
                    symbol,
                    suffix,
                    span,
                }) if let Some(symbol) = symbol.as_str().strip_prefix('-') => {
                    let symbol = Symbol::intern(symbol);
                    let integer = TokenKind::lit(token::Integer, symbol, suffix);
                    let a = tokenstream::TokenTree::token_joint_hidden(Minus, span);
                    let b = tokenstream::TokenTree::token_alone(integer, span);
                    t.push_tree(a);
                    t.push_tree(b);
                }
                TokenTree::Literal(self::Literal {
                    kind: self::LitKind::Float,
                    symbol,
                    suffix,
                    span,
                }) if let Some(symbol) = symbol.as_str().strip_prefix('-') => {
                    let symbol = Symbol::intern(symbol);
                    let float = TokenKind::lit(token::Float, symbol, suffix);
                    let a = tokenstream::TokenTree::token_joint_hidden(Minus, span);
                    let b = tokenstream::TokenTree::token_alone(float, span);
                    t.push_tree(a);
                    t.push_tree(b);
                }
                TokenTree::Literal(self::Literal { kind, symbol, suffix, span }) => {
                    t.push_tree(tokenstream::TokenTree::token_alone(
                        TokenKind::lit(kind.to_internal(), symbol, suffix),
                        span,
                    ))
                }
            }
        }

        t
    }

    pub(crate) fn ts_rustc_to_pm(&mut self, ts: tokenstream::TokenStream) -> BridgeTokenStream {
        BridgeTokenStream::new(<Vec<TokenTree<Span, Symbol>>>::from_internal((ts, self)))
    }
}

impl server::Server for &mut Rustc<'_, '_> {
    type Span = Span;
    type Symbol = Symbol;

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
        f(symbol.as_str())
    }

    fn injected_env_var(&mut self, var: &str) -> Option<String> {
        self.ecx.sess.opts.logical_env.get(var).cloned()
    }

    fn track_env_var(&mut self, var: &str, value: Option<&str>) {
        self.psess()
            .env_depinfo
            .borrow_mut()
            .insert((Symbol::intern(var), value.map(Symbol::intern)));
    }

    fn track_path(&mut self, path: &str) {
        self.psess().file_depinfo.borrow_mut().insert(Symbol::intern(path));
    }

    fn literal_from_str(&mut self, s: &str) -> Result<Literal<Self::Span, Self::Symbol>, String> {
        let name = FileName::proc_macro_source_code(s);

        let mut parser =
            new_parser_from_source_str(self.psess(), name, s.to_owned(), StripTokens::Nothing)
                .map_err(cancel_diags_into_string)?;

        let first_span = parser.token.span.data();
        let minus_present = parser.eat(exp!(Minus));

        let lit_span = parser.token.span.data();
        let token::Literal(mut lit) = parser.token.kind else {
            return Err("not a literal".to_string());
        };

        // Check no comment or whitespace surrounding the (possibly negative)
        // literal, or more tokens after it.
        if (lit_span.hi.0 - first_span.lo.0) as usize != s.len() {
            return Err("comment or whitespace around literal".to_string());
        }

        if minus_present {
            // If minus is present, check no comment or whitespace in between it
            // and the literal token.
            if first_span.hi.0 != lit_span.lo.0 {
                return Err("comment or whitespace after minus".to_string());
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
                | token::LitKind::Err(_) => {
                    return Err("non-numeric literal may not be negated".to_string());
                }
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
        let message = rustc_errors::DiagMessage::from(diagnostic.message);
        let mut diag: Diag<'_, ()> =
            Diag::new(self.psess().dcx(), diagnostic.level.to_internal(), message);
        diag.span(MultiSpan::from_spans(diagnostic.spans));
        for child in diagnostic.children {
            diag.sub(child.level.to_internal(), child.message, MultiSpan::from_spans(child.spans));
        }
        diag.emit();
    }

    fn ts_from_str(&mut self, src: &str) -> Result<BridgeTokenStream, String> {
        Ok(self.ts_rustc_to_pm(
            source_str_to_stream(
                self.psess(),
                FileName::proc_macro_source_code(src),
                src.to_string(),
                Some(self.call_site),
            )
            .map_err(cancel_diags_into_string)?,
        ))
    }

    fn ts_to_string(&mut self, stream: BridgeTokenStream) -> String {
        pprust::tts_to_string(&self.ts_pm_to_rustc(stream))
    }

    fn ts_expand_expr(&mut self, stream: BridgeTokenStream) -> Result<BridgeTokenStream, ()> {
        // Parse the expression from our tokenstream.
        let expr = try {
            let stream = self.ts_pm_to_rustc(stream);
            let mut p = Parser::new(self.psess(), stream, Some("proc_macro expand expr"));
            let expr = p.parse_expr()?;
            if p.token != token::Eof {
                p.unexpected()?;
            }
            expr
        };
        let expr = expr.map_err(|err| {
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
            ast::ExprKind::Lit(token_lit) if token_lit.kind == token::Bool => Ok(self
                .ts_rustc_to_pm(tokenstream::TokenStream::token_alone(
                    token::Ident(token_lit.symbol, IdentIsRaw::No),
                    expr.span,
                ))),
            ast::ExprKind::Lit(token_lit) => Ok(self.ts_rustc_to_pm(
                tokenstream::TokenStream::token_alone(token::Literal(*token_lit), expr.span),
            )),
            ast::ExprKind::IncludedBytes(byte_sym) => {
                let lit = token::Lit::new(
                    token::ByteStr,
                    escape_byte_str_symbol(byte_sym.as_byte_str()),
                    None,
                );
                Ok(self.ts_rustc_to_pm(tokenstream::TokenStream::token_alone(
                    token::TokenKind::Literal(lit),
                    expr.span,
                )))
            }
            ast::ExprKind::Unary(ast::UnOp::Neg, e) => match &e.kind {
                ast::ExprKind::Lit(token_lit) => match token_lit {
                    token::Lit { kind: token::Integer | token::Float, .. } => {
                        Ok(self.ts_rustc_to_pm(tokenstream::TokenStream::from_iter([
                            // FIXME: The span of the `-` token is lost when
                            // parsing, so we cannot faithfully recover it here.
                            tokenstream::TokenTree::token_joint_hidden(token::Minus, e.span),
                            tokenstream::TokenTree::token_alone(token::Literal(*token_lit), e.span),
                        ])))
                    }
                    _ => Err(()),
                },
                _ => Err(()),
            },
            _ => Err(()),
        }
    }

    fn span_debug(&mut self, span: Self::Span) -> String {
        if self.ecx.ecfg.span_debug {
            format!("{span:?}")
        } else {
            format!("{:?} bytes({}..{})", span.ctxt(), span.lo().0, span.hi().0)
        }
    }

    fn span_file(&mut self, span: Self::Span) -> String {
        self.psess()
            .source_map()
            .lookup_char_pos(span.lo())
            .file
            .name
            .prefer_remapped_unconditionally()
            .to_string()
    }

    fn span_local_file(&mut self, span: Self::Span) -> Option<String> {
        self.psess()
            .source_map()
            .lookup_char_pos(span.lo())
            .file
            .name
            .clone()
            .into_local_path()
            .map(|p| {
                p.to_str()
                    .expect("non-UTF8 file path in `proc_macro::SourceFile::path`")
                    .to_string()
            })
    }

    fn span_parent(&mut self, span: Self::Span) -> Option<Self::Span> {
        span.parent_callsite()
    }

    fn span_source(&mut self, span: Self::Span) -> Self::Span {
        span.source_callsite()
    }

    fn span_byte_range(&mut self, span: Self::Span) -> Range<usize> {
        let source_map = self.psess().source_map();

        let relative_start_pos = source_map.lookup_byte_offset(span.lo()).pos;
        let relative_end_pos = source_map.lookup_byte_offset(span.hi()).pos;

        Range { start: relative_start_pos.0 as usize, end: relative_end_pos.0 as usize }
    }
    fn span_start(&mut self, span: Self::Span) -> Self::Span {
        span.shrink_to_lo()
    }

    fn span_end(&mut self, span: Self::Span) -> Self::Span {
        span.shrink_to_hi()
    }

    fn span_line(&mut self, span: Self::Span) -> usize {
        let loc = self.psess().source_map().lookup_char_pos(span.lo());
        loc.line
    }

    fn span_column(&mut self, span: Self::Span) -> usize {
        let loc = self.psess().source_map().lookup_char_pos(span.lo());
        loc.col.to_usize() + 1
    }

    fn span_join(&mut self, first: Self::Span, second: Self::Span) -> Option<Self::Span> {
        let self_loc = self.psess().source_map().lookup_char_pos(first.lo());
        let other_loc = self.psess().source_map().lookup_char_pos(second.lo());

        if self_loc.file.stable_id != other_loc.file.stable_id {
            return None;
        }

        Some(first.to(second))
    }

    fn span_subspan(
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

    fn span_resolved_at(&mut self, span: Self::Span, at: Self::Span) -> Self::Span {
        span.with_ctxt(at.ctxt())
    }

    fn span_source_text(&mut self, span: Self::Span) -> Option<String> {
        self.psess().source_map().span_to_snippet(span).ok()
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
    fn span_save_span(&mut self, span: Self::Span) -> usize {
        self.psess().save_proc_macro_span(span)
    }

    fn span_recover_proc_macro_span(&mut self, id: usize) -> Self::Span {
        let (resolver, krate, def_site) = (&*self.ecx.resolver, self.krate, self.def_site);
        *self.rebased_spans.entry(id).or_insert_with(|| {
            // FIXME: `SyntaxContext` for spans from proc macro crates is lost during encoding,
            // replace it with a def-site context until we are encoding it properly.
            resolver.get_proc_macro_quoted_span(krate, id).with_ctxt(def_site.ctxt())
        })
    }

    fn symbol_normalize_and_validate_ident(&mut self, string: &str) -> Result<Self::Symbol, ()> {
        let sym = nfc_normalize(string);
        if rustc_lexer::is_ident(sym.as_str()) { Ok(sym) } else { Err(()) }
    }
}
