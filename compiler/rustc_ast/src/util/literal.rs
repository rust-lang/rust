//! Code related to parsing literals.

use crate::ast::{self, Lit, LitKind};
use crate::token::{self, Token};

use rustc_lexer::unescape::{unescape_byte, unescape_char};
use rustc_lexer::unescape::{unescape_byte_literal, unescape_literal, Mode};
use rustc_span::symbol::{kw, sym, Symbol};
use rustc_span::Span;

use std::ascii;
use tracing::debug;

pub enum LitError {
    NotLiteral,
    LexerError,
    InvalidSuffix,
    InvalidIntSuffix,
    InvalidFloatSuffix,
    NonDecimalFloat(u32),
    IntTooLarge,
}

impl LitKind {
    /// Converts literal token into a semantic literal.
    pub fn from_token_lit(lit: token::Lit) -> Result<LitKind, LitError> {
        let token::Lit { kind, symbol, suffix } = lit;
        if suffix.is_some() && !kind.may_have_suffix() {
            return Err(LitError::InvalidSuffix);
        }

        Ok(match kind {
            token::Bool => {
                assert!(symbol.is_bool_lit());
                LitKind::Bool(symbol == kw::True)
            }
            token::Byte => {
                return unescape_byte(symbol.as_str())
                    .map(LitKind::Byte)
                    .map_err(|_| LitError::LexerError);
            }
            token::Char => {
                return unescape_char(symbol.as_str())
                    .map(LitKind::Char)
                    .map_err(|_| LitError::LexerError);
            }

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
                let symbol = if s.contains(&['\\', '\r']) {
                    let mut buf = String::with_capacity(s.len());
                    let mut error = Ok(());
                    // Force-inlining here is aggressive but the closure is
                    // called on every char in the string, so it can be
                    // hot in programs with many long strings.
                    unescape_literal(
                        &s,
                        Mode::Str,
                        &mut #[inline(always)]
                        |_, unescaped_char| match unescaped_char {
                            Ok(c) => buf.push(c),
                            Err(err) => {
                                if err.is_fatal() {
                                    error = Err(LitError::LexerError);
                                }
                            }
                        },
                    );
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
                let symbol =
                    if s.contains('\r') {
                        let mut buf = String::with_capacity(s.len());
                        let mut error = Ok(());
                        unescape_literal(&s, Mode::RawStr, &mut |_, unescaped_char| {
                            match unescaped_char {
                                Ok(c) => buf.push(c),
                                Err(err) => {
                                    if err.is_fatal() {
                                        error = Err(LitError::LexerError);
                                    }
                                }
                            }
                        });
                        error?;
                        Symbol::intern(&buf)
                    } else {
                        symbol
                    };
                LitKind::Str(symbol, ast::StrStyle::Raw(n))
            }
            token::ByteStr => {
                let s = symbol.as_str();
                let mut buf = Vec::with_capacity(s.len());
                let mut error = Ok(());
                unescape_byte_literal(&s, Mode::ByteStr, &mut |_, unescaped_byte| {
                    match unescaped_byte {
                        Ok(c) => buf.push(c),
                        Err(err) => {
                            if err.is_fatal() {
                                error = Err(LitError::LexerError);
                            }
                        }
                    }
                });
                error?;
                LitKind::ByteStr(buf.into())
            }
            token::ByteStrRaw(_) => {
                let s = symbol.as_str();
                let bytes = if s.contains('\r') {
                    let mut buf = Vec::with_capacity(s.len());
                    let mut error = Ok(());
                    unescape_byte_literal(&s, Mode::RawByteStr, &mut |_, unescaped_byte| {
                        match unescaped_byte {
                            Ok(c) => buf.push(c),
                            Err(err) => {
                                if err.is_fatal() {
                                    error = Err(LitError::LexerError);
                                }
                            }
                        }
                    });
                    error?;
                    buf
                } else {
                    symbol.to_string().into_bytes()
                };

                LitKind::ByteStr(bytes.into())
            }
            token::Err => LitKind::Err,
        })
    }

    /// Attempts to recover a token from semantic literal.
    /// This function is used when the original token doesn't exist (e.g. the literal is created
    /// by an AST-based macro) or unavailable (e.g. from HIR pretty-printing).
    pub fn to_token_lit(&self) -> token::Lit {
        let (kind, symbol, suffix) = match *self {
            LitKind::Str(symbol, ast::StrStyle::Cooked) => {
                // Don't re-intern unless the escaped string is different.
                let s = symbol.as_str();
                let escaped = s.escape_default().to_string();
                let symbol = if s == escaped { symbol } else { Symbol::intern(&escaped) };
                (token::Str, symbol, None)
            }
            LitKind::Str(symbol, ast::StrStyle::Raw(n)) => (token::StrRaw(n), symbol, None),
            LitKind::ByteStr(ref bytes) => {
                let string = bytes
                    .iter()
                    .cloned()
                    .flat_map(ascii::escape_default)
                    .map(Into::<char>::into)
                    .collect::<String>();
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
                    ast::LitIntType::Unsigned(ty) => Some(ty.name()),
                    ast::LitIntType::Signed(ty) => Some(ty.name()),
                    ast::LitIntType::Unsuffixed => None,
                };
                (token::Integer, sym::integer(n), suffix)
            }
            LitKind::Float(symbol, ty) => {
                let suffix = match ty {
                    ast::LitFloatType::Suffixed(ty) => Some(ty.name()),
                    ast::LitFloatType::Unsuffixed => None,
                };
                (token::Float, symbol, suffix)
            }
            LitKind::Bool(value) => {
                let symbol = if value { kw::True } else { kw::False };
                (token::Bool, symbol, None)
            }
            // This only shows up in places like `-Zunpretty=hir` output, so we
            // don't bother to produce something useful.
            LitKind::Err => (token::Err, Symbol::intern("<bad-literal>"), None),
        };

        token::Lit::new(kind, symbol, suffix)
    }
}

impl Lit {
    /// Converts literal token into an AST literal.
    pub fn from_token_lit(token_lit: token::Lit, span: Span) -> Result<Lit, LitError> {
        Ok(Lit { token_lit, kind: LitKind::from_token_lit(token_lit)?, span })
    }

    /// Converts arbitrary token into an AST literal.
    ///
    /// Keep this in sync with `Token::can_begin_literal_or_bool` excluding unary negation.
    pub fn from_token(token: &Token) -> Result<Lit, LitError> {
        let lit = match token.uninterpolate().kind {
            token::Ident(name, false) if name.is_bool_lit() => {
                token::Lit::new(token::Bool, name, None)
            }
            token::Literal(lit) => lit,
            token::Interpolated(ref nt) => {
                if let token::NtExpr(expr) | token::NtLiteral(expr) = &**nt
                    && let ast::ExprKind::Lit(lit) = &expr.kind
                {
                    return Ok(lit.clone());
                }
                return Err(LitError::NotLiteral);
            }
            _ => return Err(LitError::NotLiteral),
        };

        Lit::from_token_lit(lit, token.span)
    }

    /// Attempts to recover an AST literal from semantic literal.
    /// This function is used when the original token doesn't exist (e.g. the literal is created
    /// by an AST-based macro) or unavailable (e.g. from HIR pretty-printing).
    pub fn from_lit_kind(kind: LitKind, span: Span) -> Lit {
        Lit { token_lit: kind.to_token_lit(), kind, span }
    }

    /// Losslessly convert an AST literal into a token.
    pub fn to_token(&self) -> Token {
        let kind = match self.token_lit.kind {
            token::Bool => token::Ident(self.token_lit.symbol, false),
            _ => token::Literal(self.token_lit),
        };
        Token::new(kind, self.span)
    }
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

fn filtered_float_lit(
    symbol: Symbol,
    suffix: Option<Symbol>,
    base: u32,
) -> Result<LitKind, LitError> {
    debug!("filtered_float_lit: {:?}, {:?}, {:?}", symbol, suffix, base);
    if base != 10 {
        return Err(LitError::NonDecimalFloat(base));
    }
    Ok(match suffix {
        Some(suf) => LitKind::Float(
            symbol,
            ast::LitFloatType::Suffixed(match suf {
                sym::f32 => ast::FloatTy::F32,
                sym::f64 => ast::FloatTy::F64,
                _ => return Err(LitError::InvalidFloatSuffix),
            }),
        ),
        None => LitKind::Float(symbol, ast::LitFloatType::Unsuffixed),
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

    let base = match s.as_bytes() {
        [b'0', b'x', ..] => 16,
        [b'0', b'o', ..] => 8,
        [b'0', b'b', ..] => 2,
        _ => 10,
    };

    let ty = match suffix {
        Some(suf) => match suf {
            sym::isize => ast::LitIntType::Signed(ast::IntTy::Isize),
            sym::i8 => ast::LitIntType::Signed(ast::IntTy::I8),
            sym::i16 => ast::LitIntType::Signed(ast::IntTy::I16),
            sym::i32 => ast::LitIntType::Signed(ast::IntTy::I32),
            sym::i64 => ast::LitIntType::Signed(ast::IntTy::I64),
            sym::i128 => ast::LitIntType::Signed(ast::IntTy::I128),
            sym::usize => ast::LitIntType::Unsigned(ast::UintTy::Usize),
            sym::u8 => ast::LitIntType::Unsigned(ast::UintTy::U8),
            sym::u16 => ast::LitIntType::Unsigned(ast::UintTy::U16),
            sym::u32 => ast::LitIntType::Unsigned(ast::UintTy::U32),
            sym::u64 => ast::LitIntType::Unsigned(ast::UintTy::U64),
            sym::u128 => ast::LitIntType::Unsigned(ast::UintTy::U128),
            // `1f64` and `2f32` etc. are valid float literals, and
            // `fxxx` looks more like an invalid float literal than invalid integer literal.
            _ if suf.as_str().starts_with('f') => return filtered_float_lit(symbol, suffix, base),
            _ => return Err(LitError::InvalidIntSuffix),
        },
        _ => ast::LitIntType::Unsuffixed,
    };

    let s = &s[if base != 10 { 2 } else { 0 }..];
    u128::from_str_radix(s, base).map(|i| LitKind::Int(i, ty)).map_err(|_| {
        // Small bases are lexed as if they were base 10, e.g, the string
        // might be `0b10201`. This will cause the conversion above to fail,
        // but these kinds of errors are already reported by the lexer.
        let from_lexer =
            base < 10 && s.chars().any(|c| c.to_digit(10).map_or(false, |d| d >= base));
        if from_lexer { LitError::LexerError } else { LitError::IntTooLarge }
    })
}
