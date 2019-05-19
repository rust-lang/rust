//! Code related to parsing literals.

use crate::ast::{self, Ident, Lit, LitKind};
use crate::parse::parser::Parser;
use crate::parse::PResult;
use crate::parse::token::{self, Token};
use crate::parse::unescape::{unescape_str, unescape_char, unescape_byte_str, unescape_byte};
use crate::print::pprust;
use crate::symbol::{kw, sym, Symbol};
use crate::tokenstream::{TokenStream, TokenTree};

use errors::{Applicability, Handler};
use log::debug;
use rustc_data_structures::sync::Lrc;
use syntax_pos::Span;

use std::ascii;

crate enum LitError {
    NotLiteral,
    LexerError,
    InvalidSuffix,
    InvalidIntSuffix,
    InvalidFloatSuffix,
    NonDecimalFloat(u32),
    IntTooLarge,
}

impl LitError {
    fn report(&self, diag: &Handler, lit: token::Lit, span: Span) {
        let token::Lit { kind, suffix, .. } = lit;
        match *self {
            // `NotLiteral` is not an error by itself, so we don't report
            // it and give the parser opportunity to try something else.
            LitError::NotLiteral => {}
            // `LexerError` *is* an error, but it was already reported
            // by lexer, so here we don't report it the second time.
            LitError::LexerError => {}
            LitError::InvalidSuffix => {
                expect_no_suffix(
                    diag, span, &format!("{} {} literal", kind.article(), kind.descr()), suffix
                );
            }
            LitError::InvalidIntSuffix => {
                let suf = suffix.expect("suffix error with no suffix").as_str();
                if looks_like_width_suffix(&['i', 'u'], &suf) {
                    // If it looks like a width, try to be helpful.
                    let msg = format!("invalid width `{}` for integer literal", &suf[1..]);
                    diag.struct_span_err(span, &msg)
                        .help("valid widths are 8, 16, 32, 64 and 128")
                        .emit();
                } else {
                    let msg = format!("invalid suffix `{}` for integer literal", suf);
                    diag.struct_span_err(span, &msg)
                        .span_label(span, format!("invalid suffix `{}`", suf))
                        .help("the suffix must be one of the integral types (`u32`, `isize`, etc)")
                        .emit();
                }
            }
            LitError::InvalidFloatSuffix => {
                let suf = suffix.expect("suffix error with no suffix").as_str();
                if looks_like_width_suffix(&['f'], &suf) {
                    // If it looks like a width, try to be helpful.
                    let msg = format!("invalid width `{}` for float literal", &suf[1..]);
                    diag.struct_span_err(span, &msg)
                        .help("valid widths are 32 and 64")
                        .emit();
                } else {
                    let msg = format!("invalid suffix `{}` for float literal", suf);
                    diag.struct_span_err(span, &msg)
                        .span_label(span, format!("invalid suffix `{}`", suf))
                        .help("valid suffixes are `f32` and `f64`")
                        .emit();
                }
            }
            LitError::NonDecimalFloat(base) => {
                let descr = match base {
                    16 => "hexadecimal",
                    8 => "octal",
                    2 => "binary",
                    _ => unreachable!(),
                };
                diag.struct_span_err(span, &format!("{} float literal is not supported", descr))
                    .span_label(span, "not supported")
                    .emit();
            }
            LitError::IntTooLarge => {
                diag.struct_span_err(span, "integer literal is too large")
                    .emit();
            }
        }
    }
}

impl LitKind {
    /// Converts literal token into a semantic literal.
    fn from_lit_token(lit: token::Lit) -> Result<LitKind, LitError> {
        let token::Lit { kind, symbol, suffix } = lit;
        if suffix.is_some() && !kind.may_have_suffix() {
            return Err(LitError::InvalidSuffix);
        }

        Ok(match kind {
            token::Bool => {
                assert!(symbol == kw::True || symbol == kw::False);
                LitKind::Bool(symbol == kw::True)
            }
            token::Byte => return unescape_byte(&symbol.as_str())
                .map(LitKind::Byte).map_err(|_| LitError::LexerError),
            token::Char => return unescape_char(&symbol.as_str())
                .map(LitKind::Char).map_err(|_| LitError::LexerError),

            // There are some valid suffixes for integer and float literals,
            // so all the handling is done internally.
            token::Integer => return integer_lit(symbol, suffix),
            token::Float => return float_lit(symbol, suffix),

            token::Str => {
                // If there are no characters requiring special treatment we can
                // reuse the symbol from the token. Otherwise, we must generate a
                // new symbol because the string in the LitKind is different to the
                // string in the token.
                let s = symbol.as_str();
                let symbol = if s.contains(&['\\', '\r'][..]) {
                    let mut buf = String::with_capacity(s.len());
                    let mut error = Ok(());
                    unescape_str(&s, &mut |_, unescaped_char| {
                        match unescaped_char {
                            Ok(c) => buf.push(c),
                            Err(_) => error = Err(LitError::LexerError),
                        }
                    });
                    error?;
                    Symbol::intern(&buf)
                } else {
                    symbol
                };
                LitKind::Str(symbol, ast::StrStyle::Cooked)
            }
            token::StrRaw(n) => {
                // Ditto.
                let s = symbol.as_str();
                let symbol = if s.contains('\r') {
                    Symbol::intern(&raw_str_lit(&s))
                } else {
                    symbol
                };
                LitKind::Str(symbol, ast::StrStyle::Raw(n))
            }
            token::ByteStr => {
                let s = symbol.as_str();
                let mut buf = Vec::with_capacity(s.len());
                let mut error = Ok(());
                unescape_byte_str(&s, &mut |_, unescaped_byte| {
                    match unescaped_byte {
                        Ok(c) => buf.push(c),
                        Err(_) => error = Err(LitError::LexerError),
                    }
                });
                error?;
                buf.shrink_to_fit();
                LitKind::ByteStr(Lrc::new(buf))
            }
            token::ByteStrRaw(_) => LitKind::ByteStr(Lrc::new(symbol.to_string().into_bytes())),
            token::Err => LitKind::Err(symbol),
        })
    }

    /// Attempts to recover a token from semantic literal.
    /// This function is used when the original token doesn't exist (e.g. the literal is created
    /// by an AST-based macro) or unavailable (e.g. from HIR pretty-printing).
    pub fn to_lit_token(&self) -> token::Lit {
        let (kind, symbol, suffix) = match *self {
            LitKind::Str(string, ast::StrStyle::Cooked) => {
                let escaped = string.as_str().escape_default().to_string();
                (token::Str, Symbol::intern(&escaped), None)
            }
            LitKind::Str(string, ast::StrStyle::Raw(n)) => {
                (token::StrRaw(n), string, None)
            }
            LitKind::ByteStr(ref bytes) => {
                let string = bytes.iter().cloned().flat_map(ascii::escape_default)
                    .map(Into::<char>::into).collect::<String>();
                (token::ByteStr, Symbol::intern(&string), None)
            }
            LitKind::Byte(byte) => {
                let string: String = ascii::escape_default(byte).map(Into::<char>::into).collect();
                (token::Byte, Symbol::intern(&string), None)
            }
            LitKind::Char(ch) => {
                let string: String = ch.escape_default().map(Into::<char>::into).collect();
                (token::Char, Symbol::intern(&string), None)
            }
            LitKind::Int(n, ty) => {
                let suffix = match ty {
                    ast::LitIntType::Unsigned(ty) => Some(Symbol::intern(ty.ty_to_string())),
                    ast::LitIntType::Signed(ty) => Some(Symbol::intern(ty.ty_to_string())),
                    ast::LitIntType::Unsuffixed => None,
                };
                (token::Integer, Symbol::intern(&n.to_string()), suffix)
            }
            LitKind::Float(symbol, ty) => {
                (token::Float, symbol, Some(Symbol::intern(ty.ty_to_string())))
            }
            LitKind::FloatUnsuffixed(symbol) => {
                (token::Float, symbol, None)
            }
            LitKind::Bool(value) => {
                let symbol = if value { kw::True } else { kw::False };
                (token::Bool, symbol, None)
            }
            LitKind::Err(symbol) => {
                (token::Err, symbol, None)
            }
        };

        token::Lit::new(kind, symbol, suffix)
    }
}

impl Lit {
    /// Converts literal token into an AST literal.
    fn from_lit_token(token: token::Lit, span: Span) -> Result<Lit, LitError> {
        Ok(Lit { token, node: LitKind::from_lit_token(token)?, span })
    }

    /// Converts arbitrary token into an AST literal.
    crate fn from_token(token: &Token, span: Span) -> Result<Lit, LitError> {
        let lit = match *token {
            token::Ident(ident, false) if ident.name == kw::True || ident.name == kw::False =>
                token::Lit::new(token::Bool, ident.name, None),
            token::Literal(lit) =>
                lit,
            token::Interpolated(ref nt) => {
                if let token::NtExpr(expr) | token::NtLiteral(expr) = &**nt {
                    if let ast::ExprKind::Lit(lit) = &expr.node {
                        return Ok(lit.clone());
                    }
                }
                return Err(LitError::NotLiteral);
            }
            _ => return Err(LitError::NotLiteral)
        };

        Lit::from_lit_token(lit, span)
    }

    /// Attempts to recover an AST literal from semantic literal.
    /// This function is used when the original token doesn't exist (e.g. the literal is created
    /// by an AST-based macro) or unavailable (e.g. from HIR pretty-printing).
    pub fn from_lit_kind(node: LitKind, span: Span) -> Lit {
        Lit { token: node.to_lit_token(), node, span }
    }

    /// Losslessly convert an AST literal into a token stream.
    crate fn tokens(&self) -> TokenStream {
        let token = match self.token.kind {
            token::Bool => token::Ident(Ident::new(self.token.symbol, self.span), false),
            _ => token::Literal(self.token),
        };
        TokenTree::Token(self.span, token).into()
    }
}

impl<'a> Parser<'a> {
    /// Matches `lit = true | false | token_lit`.
    crate fn parse_lit(&mut self) -> PResult<'a, Lit> {
        let mut recovered = None;
        if self.token == token::Dot {
            // Attempt to recover `.4` as `0.4`.
            recovered = self.look_ahead(1, |t| {
                if let token::Literal(token::Lit { kind: token::Integer, symbol, suffix }) = *t {
                    let next_span = self.look_ahead_span(1);
                    if self.span.hi() == next_span.lo() {
                        let s = String::from("0.") + &symbol.as_str();
                        let token = Token::lit(token::Float, Symbol::intern(&s), suffix);
                        return Some((token, self.span.to(next_span)));
                    }
                }
                None
            });
            if let Some((ref token, span)) = recovered {
                self.bump();
                self.diagnostic()
                    .struct_span_err(span, "float literals must have an integer part")
                    .span_suggestion(
                        span,
                        "must have an integer part",
                        pprust::token_to_string(&token),
                        Applicability::MachineApplicable,
                    )
                    .emit();
            }
        }

        let (token, span) = recovered.as_ref().map_or((&self.token, self.span),
                                                      |(token, span)| (token, *span));

        match Lit::from_token(token, span) {
            Ok(lit) => {
                self.bump();
                Ok(lit)
            }
            Err(LitError::NotLiteral) => {
                let msg = format!("unexpected token: {}", self.this_token_descr());
                Err(self.span_fatal(span, &msg))
            }
            Err(err) => {
                let lit = token.expect_lit();
                self.bump();
                err.report(&self.sess.span_diagnostic, lit, span);
                let lit = token::Lit::new(token::Err, lit.symbol, lit.suffix);
                Lit::from_lit_token(lit, span).map_err(|_| unreachable!())
            }
        }
    }
}

crate fn expect_no_suffix(diag: &Handler, sp: Span, kind: &str, suffix: Option<Symbol>) {
    if let Some(suf) = suffix {
        let mut err = if kind == "a tuple index" &&
                         [sym::i32, sym::u32, sym::isize, sym::usize].contains(&suf) {
            // #59553: warn instead of reject out of hand to allow the fix to percolate
            // through the ecosystem when people fix their macros
            let mut err = diag.struct_span_warn(
                sp,
                &format!("suffixes on {} are invalid", kind),
            );
            err.note(&format!(
                "`{}` is *temporarily* accepted on tuple index fields as it was \
                    incorrectly accepted on stable for a few releases",
                suf,
            ));
            err.help(
                "on proc macros, you'll want to use `syn::Index::from` or \
                    `proc_macro::Literal::*_unsuffixed` for code that will desugar \
                    to tuple field access",
            );
            err.note(
                "for more context, see https://github.com/rust-lang/rust/issues/60210",
            );
            err
        } else {
            diag.struct_span_err(sp, &format!("suffixes on {} are invalid", kind))
        };
        err.span_label(sp, format!("invalid suffix `{}`", suf));
        err.emit();
    }
}

/// Parses a string representing a raw string literal into its final form. The
/// only operation this does is convert embedded CRLF into a single LF.
fn raw_str_lit(lit: &str) -> String {
    debug!("raw_str_lit: {:?}", lit);
    let mut res = String::with_capacity(lit.len());

    let mut chars = lit.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\r' {
            if *chars.peek().unwrap() != '\n' {
                panic!("lexer accepted bare CR");
            }
            chars.next();
            res.push('\n');
        } else {
            res.push(c);
        }
    }

    res.shrink_to_fit();
    res
}

// Checks if `s` looks like i32 or u1234 etc.
fn looks_like_width_suffix(first_chars: &[char], s: &str) -> bool {
    s.len() > 1 && s.starts_with(first_chars) && s[1..].chars().all(|c| c.is_ascii_digit())
}

fn strip_underscores(symbol: Symbol) -> Symbol {
    // Do not allocate a new string unless necessary.
    let s = symbol.as_str();
    if s.contains('_') {
        let mut s = s.to_string();
        s.retain(|c| c != '_');
        return Symbol::intern(&s);
    }
    symbol
}

fn filtered_float_lit(symbol: Symbol, suffix: Option<Symbol>, base: u32)
                      -> Result<LitKind, LitError> {
    debug!("filtered_float_lit: {:?}, {:?}, {:?}", symbol, suffix, base);
    if base != 10 {
        return Err(LitError::NonDecimalFloat(base));
    }
    Ok(match suffix {
        Some(suf) => match suf {
            sym::f32 => LitKind::Float(symbol, ast::FloatTy::F32),
            sym::f64 => LitKind::Float(symbol, ast::FloatTy::F64),
            _ => return Err(LitError::InvalidFloatSuffix),
        }
        None => LitKind::FloatUnsuffixed(symbol)
    })
}

fn float_lit(symbol: Symbol, suffix: Option<Symbol>) -> Result<LitKind, LitError> {
    debug!("float_lit: {:?}, {:?}", symbol, suffix);
    filtered_float_lit(strip_underscores(symbol), suffix, 10)
}

fn integer_lit(symbol: Symbol, suffix: Option<Symbol>) -> Result<LitKind, LitError> {
    debug!("integer_lit: {:?}, {:?}", symbol, suffix);
    let symbol = strip_underscores(symbol);
    let s = symbol.as_str();

    let mut base = 10;
    if s.len() > 1 && s.as_bytes()[0] == b'0' {
        match s.as_bytes()[1] {
            b'x' => base = 16,
            b'o' => base = 8,
            b'b' => base = 2,
            _ => {}
        }
    }

    let ty = match suffix {
        Some(suf) => match suf {
            sym::isize => ast::LitIntType::Signed(ast::IntTy::Isize),
            sym::i8  => ast::LitIntType::Signed(ast::IntTy::I8),
            sym::i16 => ast::LitIntType::Signed(ast::IntTy::I16),
            sym::i32 => ast::LitIntType::Signed(ast::IntTy::I32),
            sym::i64 => ast::LitIntType::Signed(ast::IntTy::I64),
            sym::i128 => ast::LitIntType::Signed(ast::IntTy::I128),
            sym::usize => ast::LitIntType::Unsigned(ast::UintTy::Usize),
            sym::u8  => ast::LitIntType::Unsigned(ast::UintTy::U8),
            sym::u16 => ast::LitIntType::Unsigned(ast::UintTy::U16),
            sym::u32 => ast::LitIntType::Unsigned(ast::UintTy::U32),
            sym::u64 => ast::LitIntType::Unsigned(ast::UintTy::U64),
            sym::u128 => ast::LitIntType::Unsigned(ast::UintTy::U128),
            // `1f64` and `2f32` etc. are valid float literals, and
            // `fxxx` looks more like an invalid float literal than invalid integer literal.
            _ if suf.as_str().starts_with('f') => return filtered_float_lit(symbol, suffix, base),
            _ => return Err(LitError::InvalidIntSuffix),
        }
        _ => ast::LitIntType::Unsuffixed
    };

    let s = &s[if base != 10 { 2 } else { 0 } ..];
    u128::from_str_radix(s, base).map(|i| LitKind::Int(i, ty)).map_err(|_| {
        // Small bases are lexed as if they were base 10, e.g, the string
        // might be `0b10201`. This will cause the conversion above to fail,
        // but these kinds of errors are already reported by the lexer.
        let from_lexer =
            base < 10 && s.chars().any(|c| c.to_digit(10).map_or(false, |d| d >= base));
        if from_lexer { LitError::LexerError } else { LitError::IntTooLarge }
    })
}
