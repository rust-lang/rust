//! Code related to parsing literals.

use std::{ascii, fmt, str};

use rustc_literal_escaper::{
    MixedUnit, unescape_byte, unescape_byte_str, unescape_c_str, unescape_char, unescape_str,
};
use rustc_span::{Span, Symbol, kw, sym};
use tracing::debug;

use crate::ast::{self, LitKind, MetaItemLit, StrStyle};
use crate::token::{self, Token};

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
    InvalidSuffix(Symbol),
    InvalidIntSuffix(Symbol),
    InvalidFloatSuffix(Symbol),
    NonDecimalFloat(u32), // u32 is the base
    IntTooLarge(u32),     // u32 is the base
}

impl LitKind {
    /// Converts literal token into a semantic literal.
    pub fn from_token_lit(lit: token::Lit) -> Result<LitKind, LitError> {
        let token::Lit { kind, symbol, suffix } = lit;
        if let Some(suffix) = suffix
            && !kind.may_have_suffix()
        {
            return Err(LitError::InvalidSuffix(suffix));
        }

        // For byte/char/string literals, chars and escapes have already been
        // checked in the lexer (in `cook_lexer_literal`). So we can assume all
        // chars and escapes are valid here.
        Ok(match kind {
            token::Bool => {
                assert!(symbol.is_bool_lit());
                LitKind::Bool(symbol == kw::True)
            }
            token::Byte => {
                return unescape_byte(symbol.as_str())
                    .map(LitKind::Byte)
                    .map_err(|_| panic!("failed to unescape byte literal"));
            }
            token::Char => {
                return unescape_char(symbol.as_str())
                    .map(LitKind::Char)
                    .map_err(|_| panic!("failed to unescape char literal"));
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
                // Vanilla strings are so common we optimize for the common case where no chars
                // requiring special behaviour are present.
                let symbol = if s.contains('\\') {
                    let mut buf = String::with_capacity(s.len());
                    // Force-inlining here is aggressive but the closure is
                    // called on every char in the string, so it can be hot in
                    // programs with many long strings containing escapes.
                    unescape_str(
                        s,
                        #[inline(always)]
                        |_, res| match res {
                            Ok(c) => buf.push(c),
                            Err(err) => {
                                assert!(!err.is_fatal(), "failed to unescape string literal")
                            }
                        },
                    );
                    Symbol::intern(&buf)
                } else {
                    symbol
                };
                LitKind::Str(symbol, ast::StrStyle::Cooked)
            }
            token::StrRaw(n) => {
                // Raw strings have no escapes so no work is needed here.
                LitKind::Str(symbol, ast::StrStyle::Raw(n))
            }
            token::ByteStr => {
                let s = symbol.as_str();
                let mut buf = Vec::with_capacity(s.len());
                unescape_byte_str(s, |_, res| match res {
                    Ok(b) => buf.push(b),
                    Err(err) => {
                        assert!(!err.is_fatal(), "failed to unescape string literal")
                    }
                });
                LitKind::ByteStr(buf.into(), StrStyle::Cooked)
            }
            token::ByteStrRaw(n) => {
                // Raw strings have no escapes so we can convert the symbol
                // directly to a `Arc<u8>`.
                let buf = symbol.as_str().to_owned().into_bytes();
                LitKind::ByteStr(buf.into(), StrStyle::Raw(n))
            }
            token::CStr => {
                let s = symbol.as_str();
                let mut buf = Vec::with_capacity(s.len());
                unescape_c_str(s, |_span, c| match c {
                    Ok(MixedUnit::Char(c)) => {
                        buf.extend_from_slice(c.encode_utf8(&mut [0; 4]).as_bytes())
                    }
                    Ok(MixedUnit::HighByte(b)) => buf.push(b),
                    Err(err) => {
                        assert!(!err.is_fatal(), "failed to unescape C string literal")
                    }
                });
                buf.push(0);
                LitKind::CStr(buf.into(), StrStyle::Cooked)
            }
            token::CStrRaw(n) => {
                // Raw strings have no escapes so we can convert the symbol
                // directly to a `Arc<u8>` after appending the terminating NUL
                // char.
                let mut buf = symbol.as_str().to_owned().into_bytes();
                buf.push(0);
                LitKind::CStr(buf.into(), StrStyle::Raw(n))
            }
            token::Err(guar) => LitKind::Err(guar),
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
            LitKind::Err(_) => {
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
            LitKind::Err(guar) => token::Err(guar),
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
        Some(suffix) => LitKind::Float(
            symbol,
            ast::LitFloatType::Suffixed(match suffix {
                sym::f16 => ast::FloatTy::F16,
                sym::f32 => ast::FloatTy::F32,
                sym::f64 => ast::FloatTy::F64,
                sym::f128 => ast::FloatTy::F128,
                _ => return Err(LitError::InvalidFloatSuffix(suffix)),
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
            _ => return Err(LitError::InvalidIntSuffix(suf)),
        },
        _ => ast::LitIntType::Unsuffixed,
    };

    let s = &s[if base != 10 { 2 } else { 0 }..];
    u128::from_str_radix(s, base)
        .map(|i| LitKind::Int(i.into(), ty))
        .map_err(|_| LitError::IntTooLarge(base))
}
