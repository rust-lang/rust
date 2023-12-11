//! Code related to parsing literals.

use crate::ast::{self, LitKind, MetaItemLit, StrStyle};
use crate::token::{self, Token};
use rustc_lexer::unescape::{
    byte_from_char, unescape_c_string, unescape_literal, CStrUnit, EscapeError, Mode,
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
    EscapeError {
        mode: Mode,
        // Length before the string content, e.g. 1 for "a", 5 for br##"a"##
        prefix_len: u32,
        // The range is the byte range of the bad character, using a zero index.
        range: Range<usize>,
        err: EscapeError,
    },
    InvalidSuffix,
    InvalidIntSuffix,
    InvalidFloatSuffix,
    NonDecimalFloat(u32),
    IntTooLarge(u32),
}

impl LitKind {
    /// Converts literal token into a semantic literal. The return value has
    /// two parts:
    /// - The `Result` indicates success or failure.
    /// - The `Vec` contains all found errors and warnings.
    ///
    /// If we only had to deal with errors, we could use the more obvious
    /// `Result<LitKind, Vec<LitError>>`; on failure the caller would just
    /// (optionally) print errors and take the error path and stop early. But
    /// it's possible to succeed with zero errors and one or more warnings, and
    /// in that case the caller should (optionally) print the warnings, but
    /// also proceed with a valid `LitKind`. This return type facilitates that.
    pub fn from_token_lit_with_errs(lit: token::Lit) -> (Result<LitKind, ()>, Vec<LitError>) {
        let token::Lit { kind, symbol, suffix } = lit;
        if suffix.is_some() && !kind.may_have_suffix() {
            // Note: we return a single error here. We could instead continue
            // processing, possibly returning multiple errors.
            return (Err(()), vec![LitError::InvalidSuffix]);
        }

        let mut errs = vec![];
        let mut has_fatal = false;

        let res = match kind {
            token::Bool => {
                assert!(symbol.is_bool_lit());
                Ok(LitKind::Bool(symbol == kw::True))
            }
            token::Byte => {
                let mode = Mode::Byte;
                let mut res = None;
                unescape_literal(symbol.as_str(), mode, &mut |range, unescaped_char| {
                    match unescaped_char {
                        Ok(c) => res = Some(c),
                        Err(err) => {
                            has_fatal |= err.is_fatal();
                            errs.push(LitError::EscapeError {
                                mode,
                                prefix_len: 2, // b'
                                range,
                                err,
                            });
                        }
                    }
                });
                if !has_fatal { Ok(LitKind::Byte(byte_from_char(res.unwrap()))) } else { Err(()) }
            }
            token::Char => {
                let mode = Mode::Char;
                let mut res = None;
                unescape_literal(symbol.as_str(), mode, &mut |range, unescaped_char| {
                    match unescaped_char {
                        Ok(c) => res = Some(c),
                        Err(err) => {
                            has_fatal |= err.is_fatal();
                            errs.push(LitError::EscapeError {
                                mode,
                                prefix_len: 1, // '
                                range,
                                err,
                            });
                        }
                    }
                });
                if !has_fatal { Ok(LitKind::Char(res.unwrap())) } else { Err(()) }
            }

            // There are some valid suffixes for integer and float literals,
            // so all the handling is done internally.
            token::Integer => {
                return match integer_lit(symbol, suffix) {
                    Ok(lit_kind) => (Ok(lit_kind), vec![]),
                    Err(err) => (Err(()), vec![err]),
                };
            }
            token::Float => {
                return match float_lit(symbol, suffix) {
                    Ok(lit_kind) => (Ok(lit_kind), vec![]),
                    Err(err) => (Err(()), vec![err]),
                };
            }

            token::Str => {
                // If there are no characters requiring special treatment we can
                // reuse the symbol from the token. Otherwise, we must generate a
                // new symbol because the string in the LitKind is different to the
                // string in the token.
                let mode = Mode::Str;
                let s = symbol.as_str();
                // Vanilla strings are so common we optimize for the common case where no chars
                // requiring special behaviour are present.
                if s.contains(['\\', '\r']) {
                    let mut buf = String::with_capacity(s.len());
                    // Force-inlining here is aggressive but the closure is
                    // called on every char in the string, so it can be
                    // hot in programs with many long strings.
                    unescape_literal(
                        s,
                        mode,
                        &mut #[inline(always)]
                        |range, unescaped_char| match unescaped_char {
                            Ok(c) => buf.push(c),
                            Err(err) => {
                                has_fatal |= err.is_fatal();
                                errs.push(LitError::EscapeError {
                                    mode,
                                    prefix_len: 1, // "
                                    range,
                                    err,
                                });
                            }
                        },
                    );
                    if !has_fatal {
                        Ok(LitKind::Str(Symbol::intern(&buf), ast::StrStyle::Cooked))
                    } else {
                        Err(())
                    }
                } else {
                    Ok(LitKind::Str(symbol, ast::StrStyle::Cooked))
                }
            }
            token::StrRaw(n) => {
                // Raw strings have no escapes, so we only need to check for invalid chars, and we
                // can reuse the symbol on success.
                let mode = Mode::RawStr;
                let s = symbol.as_str();
                unescape_literal(s, mode, &mut |range, unescaped_char| match unescaped_char {
                    Ok(_) => {}
                    Err(err) => {
                        has_fatal |= err.is_fatal();
                        errs.push(LitError::EscapeError {
                            mode,
                            prefix_len: 2 + n as u32, // r", r#", r##", etc.
                            range,
                            err,
                        });
                    }
                });
                if !has_fatal { Ok(LitKind::Str(symbol, ast::StrStyle::Raw(n))) } else { Err(()) }
            }
            token::ByteStr => {
                let mode = Mode::ByteStr;
                let s = symbol.as_str();
                let mut buf = Vec::with_capacity(s.len());
                unescape_literal(s, mode, &mut |range, c| match c {
                    Ok(c) => buf.push(byte_from_char(c)),
                    Err(err) => {
                        has_fatal |= err.is_fatal();
                        errs.push(LitError::EscapeError {
                            mode,
                            prefix_len: 2, // b"
                            range,
                            err,
                        });
                    }
                });
                if !has_fatal {
                    Ok(LitKind::ByteStr(buf.into(), StrStyle::Cooked))
                } else {
                    Err(())
                }
            }
            token::ByteStrRaw(n) => {
                // Raw strings have no escapes, so we only need to check for invalid chars, and we
                // can convert the symbol directly to a `Lrc<u8>` on success.
                let mode = Mode::RawByteStr;
                let s = symbol.as_str();
                unescape_literal(s, mode, &mut |range, c| match c {
                    Ok(_) => {}
                    Err(err) => {
                        has_fatal |= err.is_fatal();
                        errs.push(LitError::EscapeError {
                            mode,
                            prefix_len: 3 + n as u32, // br", br#", br##", etc.
                            range,
                            err,
                        });
                    }
                });
                if !has_fatal {
                    Ok(LitKind::ByteStr(s.to_owned().into_bytes().into(), StrStyle::Raw(n)))
                } else {
                    Err(())
                }
            }
            token::CStr => {
                let mode = Mode::CStr;
                let s = symbol.as_str();
                let mut buf = Vec::with_capacity(s.len());
                unescape_c_string(s, mode, &mut |range, c| match c {
                    Ok(CStrUnit::Byte(b)) => buf.push(b),
                    Ok(CStrUnit::Char(c)) => {
                        buf.extend_from_slice(c.encode_utf8(&mut [0; 4]).as_bytes())
                    }
                    Err(err) => {
                        has_fatal |= err.is_fatal();
                        errs.push(LitError::EscapeError {
                            mode,
                            prefix_len: 2, // c"
                            range,
                            err,
                        });
                    }
                });
                if !has_fatal {
                    buf.push(0);
                    Ok(LitKind::CStr(buf.into(), StrStyle::Cooked))
                } else {
                    Err(())
                }
            }
            token::CStrRaw(n) => {
                // Raw strings have no escapes, so we only need to check for invalid chars, and we
                // can convert the symbol directly to a `Lrc<u8>` (after appending a nul char) on
                // success.
                let mode = Mode::RawCStr;
                let s = symbol.as_str();
                unescape_c_string(s, mode, &mut |range, c| match c {
                    Ok(_) => {}
                    Err(err) => {
                        has_fatal |= err.is_fatal();
                        errs.push(LitError::EscapeError {
                            mode,
                            prefix_len: 3 + n as u32, // cr", cr#", cr##", etc.
                            range,
                            err,
                        });
                    }
                });
                if !has_fatal {
                    let mut buf = s.to_owned().into_bytes();
                    buf.push(0);
                    Ok(LitKind::CStr(buf.into(), StrStyle::Raw(n)))
                } else {
                    Err(())
                }
            }
            token::Err => Ok(LitKind::Err),
        };
        (res, errs)
    }

    // Use this one for call sites where we don't need to print error messages
    // about invalid literals.
    pub fn from_token_lit(lit: token::Lit) -> Result<LitKind, ()> {
        LitKind::from_token_lit_with_errs(lit).0
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
    /// Converts a token literal into a meta item literal. See
    /// `LitKind::from_token_lit` for an explanation of the return type.
    pub fn from_token_lit_with_errs(
        token_lit: token::Lit,
        span: Span,
    ) -> (Result<MetaItemLit, ()>, Vec<LitError>) {
        let (lit, errs) = LitKind::from_token_lit_with_errs(token_lit);
        let lit = lit.map(|kind| MetaItemLit {
            symbol: token_lit.symbol,
            suffix: token_lit.suffix,
            kind,
            span,
        });
        (lit, errs)
    }

    // Use this one for call sites where we don't need to print error messages
    // about invalid literals.
    pub fn from_token_lit(token_lit: token::Lit, span: Span) -> Result<MetaItemLit, ()> {
        MetaItemLit::from_token_lit_with_errs(token_lit, span).0
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
