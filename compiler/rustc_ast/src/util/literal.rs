//! Code related to parsing literals.

use crate::ast::{self, LitKind, MetaItemLit, StrStyle};
use crate::token::{self, Token};
use rustc_lexer::unescape::{
    byte_from_char, unescape_byte, unescape_c_string, unescape_char, unescape_literal, CStrUnit,
    Mode,
};
use rustc_span::symbol::{kw, sym, Symbol};
use rustc_span::Span;
use std::ops::Range;
use std::{ascii, fmt, str};

// Escapes a string, represented as a symbol. Reuses the original symbol,
// avoiding interning, if no changes are required.
pub fn escape_string_symbol(symbol: Symbol) -> Symbol {
    let s = symbol.as_str();
    let escaped = s.escape_default().to_string();
    if s == escaped { symbol } else { Symbol::intern(&escaped) }
}

// Escapes a char.
pub fn escape_char_symbol(ch: char) -> Symbol {
    let s: String = ch.escape_default().map(Into::<char>::into).collect();
    Symbol::intern(&s)
}

// Escapes a byte string.
pub fn escape_byte_str_symbol(bytes: &[u8]) -> Symbol {
    let s = bytes.escape_ascii().to_string();
    Symbol::intern(&s)
}

#[derive(Debug)]
pub enum LitError {
    LexerError,
    InvalidSuffix,
    InvalidIntSuffix,
    InvalidFloatSuffix,
    NonDecimalFloat(u32),
    IntTooLarge(u32),
    NulInCStr(Range<usize>),
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
                let symbol = if s.contains(['\\', '\r']) {
                    let mut buf = String::with_capacity(s.len());
                    let mut error = Ok(());
                    // Force-inlining here is aggressive but the closure is
                    // called on every char in the string, so it can be
                    // hot in programs with many long strings.
                    unescape_literal(
                        s,
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
                        unescape_literal(s, Mode::RawStr, &mut |_, unescaped_char| {
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
                unescape_literal(s, Mode::ByteStr, &mut |_, c| match c {
                    Ok(c) => buf.push(byte_from_char(c)),
                    Err(err) => {
                        if err.is_fatal() {
                            error = Err(LitError::LexerError);
                        }
                    }
                });
                error?;
                LitKind::ByteStr(buf.into(), StrStyle::Cooked)
            }
            token::ByteStrRaw(n) => {
                let s = symbol.as_str();
                let bytes = if s.contains('\r') {
                    let mut buf = Vec::with_capacity(s.len());
                    let mut error = Ok(());
                    unescape_literal(s, Mode::RawByteStr, &mut |_, c| match c {
                        Ok(c) => buf.push(byte_from_char(c)),
                        Err(err) => {
                            if err.is_fatal() {
                                error = Err(LitError::LexerError);
                            }
                        }
                    });
                    error?;
                    buf
                } else {
                    symbol.to_string().into_bytes()
                };

                LitKind::ByteStr(bytes.into(), StrStyle::Raw(n))
            }
            token::CStr => {
                let s = symbol.as_str();
                let mut buf = Vec::with_capacity(s.len());
                let mut error = Ok(());
                unescape_c_string(s, Mode::CStr, &mut |span, c| match c {
                    Ok(CStrUnit::Byte(0) | CStrUnit::Char('\0')) => {
                        error = Err(LitError::NulInCStr(span));
                    }
                    Ok(CStrUnit::Byte(b)) => buf.push(b),
                    Ok(CStrUnit::Char(c)) if c.len_utf8() == 1 => buf.push(c as u8),
                    Ok(CStrUnit::Char(c)) => {
                        buf.extend_from_slice(c.encode_utf8(&mut [0; 4]).as_bytes())
                    }
                    Err(err) => {
                        if err.is_fatal() {
                            error = Err(LitError::LexerError);
                        }
                    }
                });
                error?;
                buf.push(0);
                LitKind::CStr(buf.into(), StrStyle::Cooked)
            }
            token::CStrRaw(n) => {
                let s = symbol.as_str();
                let mut buf = Vec::with_capacity(s.len());
                let mut error = Ok(());
                unescape_c_string(s, Mode::RawCStr, &mut |span, c| match c {
                    Ok(CStrUnit::Byte(0) | CStrUnit::Char('\0')) => {
                        error = Err(LitError::NulInCStr(span));
                    }
                    Ok(CStrUnit::Byte(b)) => buf.push(b),
                    Ok(CStrUnit::Char(c)) if c.len_utf8() == 1 => buf.push(c as u8),
                    Ok(CStrUnit::Char(c)) => {
                        buf.extend_from_slice(c.encode_utf8(&mut [0; 4]).as_bytes())
                    }
                    Err(err) => {
                        if err.is_fatal() {
                            error = Err(LitError::LexerError);
                        }
                    }
                });
                error?;
                buf.push(0);
                LitKind::CStr(buf.into(), StrStyle::Raw(n))
            }
            token::Err => LitKind::Err,
        })
    }
}

impl fmt::Display for LitKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            LitKind::Byte(b) => {
                let b: String = ascii::escape_default(b).map(Into::<char>::into).collect();
                write!(f, "b'{b}'")?;
            }
            LitKind::Char(ch) => write!(f, "'{}'", escape_char_symbol(ch))?,
            LitKind::Str(sym, StrStyle::Cooked) => write!(f, "\"{}\"", escape_string_symbol(sym))?,
            LitKind::Str(sym, StrStyle::Raw(n)) => write!(
                f,
                "r{delim}\"{string}\"{delim}",
                delim = "#".repeat(n as usize),
                string = sym
            )?,
            LitKind::ByteStr(ref bytes, StrStyle::Cooked) => {
                write!(f, "b\"{}\"", escape_byte_str_symbol(bytes))?
            }
            LitKind::ByteStr(ref bytes, StrStyle::Raw(n)) => {
                // Unwrap because raw byte string literals can only contain ASCII.
                let symbol = str::from_utf8(bytes).unwrap();
                write!(
                    f,
                    "br{delim}\"{string}\"{delim}",
                    delim = "#".repeat(n as usize),
                    string = symbol
                )?;
            }
            LitKind::CStr(ref bytes, StrStyle::Cooked) => {
                write!(f, "c\"{}\"", escape_byte_str_symbol(bytes))?
            }
            LitKind::CStr(ref bytes, StrStyle::Raw(n)) => {
                // This can only be valid UTF-8.
                let symbol = str::from_utf8(bytes).unwrap();
                write!(f, "cr{delim}\"{symbol}\"{delim}", delim = "#".repeat(n as usize),)?;
            }
            LitKind::Int(n, ty) => {
                write!(f, "{n}")?;
                match ty {
                    ast::LitIntType::Unsigned(ty) => write!(f, "{}", ty.name())?,
                    ast::LitIntType::Signed(ty) => write!(f, "{}", ty.name())?,
                    ast::LitIntType::Unsuffixed => {}
                }
            }
            LitKind::Float(symbol, ty) => {
                write!(f, "{symbol}")?;
                match ty {
                    ast::LitFloatType::Suffixed(ty) => write!(f, "{}", ty.name())?,
                    ast::LitFloatType::Unsuffixed => {}
                }
            }
            LitKind::Bool(b) => write!(f, "{}", if b { "true" } else { "false" })?,
            LitKind::Err => {
                // This only shows up in places like `-Zunpretty=hir` output, so we
                // don't bother to produce something useful.
                write!(f, "<bad-literal>")?;
            }
        }

        Ok(())
    }
}

impl MetaItemLit {
    /// Converts a token literal into a meta item literal.
    pub fn from_token_lit(token_lit: token::Lit, span: Span) -> Result<MetaItemLit, LitError> {
        Ok(MetaItemLit {
            symbol: token_lit.symbol,
            suffix: token_lit.suffix,
            kind: LitKind::from_token_lit(token_lit)?,
            span,
        })
    }

    /// Cheaply converts a meta item literal into a token literal.
    pub fn as_token_lit(&self) -> token::Lit {
        let kind = match self.kind {
            LitKind::Bool(_) => token::Bool,
            LitKind::Str(_, ast::StrStyle::Cooked) => token::Str,
            LitKind::Str(_, ast::StrStyle::Raw(n)) => token::StrRaw(n),
            LitKind::ByteStr(_, ast::StrStyle::Cooked) => token::ByteStr,
            LitKind::ByteStr(_, ast::StrStyle::Raw(n)) => token::ByteStrRaw(n),
            LitKind::CStr(_, ast::StrStyle::Cooked) => token::CStr,
            LitKind::CStr(_, ast::StrStyle::Raw(n)) => token::CStrRaw(n),
            LitKind::Byte(_) => token::Byte,
            LitKind::Char(_) => token::Char,
            LitKind::Int(..) => token::Integer,
            LitKind::Float(..) => token::Float,
            LitKind::Err => token::Err,
        };

        token::Lit::new(kind, self.symbol, self.suffix)
    }

    /// Converts an arbitrary token into meta item literal.
    pub fn from_token(token: &Token) -> Option<MetaItemLit> {
        token::Lit::from_token(token)
            .and_then(|token_lit| MetaItemLit::from_token_lit(token_lit, token.span).ok())
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
        let from_lexer = base < 10 && s.chars().any(|c| c.to_digit(10).is_some_and(|d| d >= base));
        if from_lexer { LitError::LexerError } else { LitError::IntTooLarge(base) }
    })
}
