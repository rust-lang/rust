//! Code related to parsing literals.

use crate::ast::{self, Ident, Lit, LitKind};
use crate::parse::parser::Parser;
use crate::parse::PResult;
use crate::parse::token::{self, Token};
use crate::parse::unescape::{unescape_str, unescape_char, unescape_byte_str, unescape_byte};
use crate::print::pprust;
use crate::symbol::{keywords, Symbol};
use crate::tokenstream::{TokenStream, TokenTree};

use errors::{Applicability, Handler};
use log::debug;
use rustc_data_structures::sync::Lrc;
use syntax_pos::Span;

use std::ascii;

macro_rules! err {
    ($opt_diag:expr, |$span:ident, $diag:ident| $($body:tt)*) => {
        match $opt_diag {
            Some(($span, $diag)) => { $($body)* }
            None => return None,
        }
    }
}

impl LitKind {
    /// Converts literal token with a suffix into a semantic literal.
    /// Works speculatively and may return `None` if diagnostic handler is not passed.
    /// If diagnostic handler is passed, always returns `Some`,
    /// possibly after reporting non-fatal errors and recovery.
    fn from_lit_token(
        lit: token::Lit,
        suf: Option<Symbol>,
        diag: Option<(Span, &Handler)>
    ) -> Option<LitKind> {
        if suf.is_some() && !lit.may_have_suffix() {
            err!(diag, |span, diag| {
                expect_no_suffix(span, diag, &format!("a {}", lit.literal_name()), suf)
            });
        }

        Some(match lit {
            token::Bool(i) => {
                assert!(i == keywords::True.name() || i == keywords::False.name());
                LitKind::Bool(i == keywords::True.name())
            }
            token::Byte(i) => {
                match unescape_byte(&i.as_str()) {
                    Ok(c) => LitKind::Byte(c),
                    Err(_) => LitKind::Err(i),
                }
            },
            token::Char(i) => {
                match unescape_char(&i.as_str()) {
                    Ok(c) => LitKind::Char(c),
                    Err(_) => LitKind::Err(i),
                }
            },
            token::Err(i) => LitKind::Err(i),

            // There are some valid suffixes for integer and float literals,
            // so all the handling is done internally.
            token::Integer(s) => return integer_lit(&s.as_str(), suf, diag),
            token::Float(s) => return float_lit(&s.as_str(), suf, diag),

            token::Str_(mut sym) => {
                // If there are no characters requiring special treatment we can
                // reuse the symbol from the Token. Otherwise, we must generate a
                // new symbol because the string in the LitKind is different to the
                // string in the Token.
                let mut has_error = false;
                let s = &sym.as_str();
                if s.as_bytes().iter().any(|&c| c == b'\\' || c == b'\r') {
                    let mut buf = String::with_capacity(s.len());
                    unescape_str(s, &mut |_, unescaped_char| {
                        match unescaped_char {
                            Ok(c) => buf.push(c),
                            Err(_) => has_error = true,
                        }
                    });
                    if has_error {
                        return Some(LitKind::Err(sym));
                    }
                    sym = Symbol::intern(&buf)
                }

                LitKind::Str(sym, ast::StrStyle::Cooked)
            }
            token::StrRaw(mut sym, n) => {
                // Ditto.
                let s = &sym.as_str();
                if s.contains('\r') {
                    sym = Symbol::intern(&raw_str_lit(s));
                }
                LitKind::Str(sym, ast::StrStyle::Raw(n))
            }
            token::ByteStr(i) => {
                let s = &i.as_str();
                let mut buf = Vec::with_capacity(s.len());
                let mut has_error = false;
                unescape_byte_str(s, &mut |_, unescaped_byte| {
                    match unescaped_byte {
                        Ok(c) => buf.push(c),
                        Err(_) => has_error = true,
                    }
                });
                if has_error {
                    return Some(LitKind::Err(i));
                }
                buf.shrink_to_fit();
                LitKind::ByteStr(Lrc::new(buf))
            }
            token::ByteStrRaw(i, _) => {
                LitKind::ByteStr(Lrc::new(i.to_string().into_bytes()))
            }
        })
    }

    /// Attempts to recover a token from semantic literal.
    /// This function is used when the original token doesn't exist (e.g. the literal is created
    /// by an AST-based macro) or unavailable (e.g. from HIR pretty-printing).
    pub fn to_lit_token(&self) -> (token::Lit, Option<Symbol>) {
        match *self {
            LitKind::Str(string, ast::StrStyle::Cooked) => {
                let escaped = string.as_str().escape_default().to_string();
                (token::Lit::Str_(Symbol::intern(&escaped)), None)
            }
            LitKind::Str(string, ast::StrStyle::Raw(n)) => {
                (token::Lit::StrRaw(string, n), None)
            }
            LitKind::ByteStr(ref bytes) => {
                let string = bytes.iter().cloned().flat_map(ascii::escape_default)
                    .map(Into::<char>::into).collect::<String>();
                (token::Lit::ByteStr(Symbol::intern(&string)), None)
            }
            LitKind::Byte(byte) => {
                let string: String = ascii::escape_default(byte).map(Into::<char>::into).collect();
                (token::Lit::Byte(Symbol::intern(&string)), None)
            }
            LitKind::Char(ch) => {
                let string: String = ch.escape_default().map(Into::<char>::into).collect();
                (token::Lit::Char(Symbol::intern(&string)), None)
            }
            LitKind::Int(n, ty) => {
                let suffix = match ty {
                    ast::LitIntType::Unsigned(ty) => Some(Symbol::intern(ty.ty_to_string())),
                    ast::LitIntType::Signed(ty) => Some(Symbol::intern(ty.ty_to_string())),
                    ast::LitIntType::Unsuffixed => None,
                };
                (token::Lit::Integer(Symbol::intern(&n.to_string())), suffix)
            }
            LitKind::Float(symbol, ty) => {
                (token::Lit::Float(symbol), Some(Symbol::intern(ty.ty_to_string())))
            }
            LitKind::FloatUnsuffixed(symbol) => (token::Lit::Float(symbol), None),
            LitKind::Bool(value) => {
                let kw = if value { keywords::True } else { keywords::False };
                (token::Lit::Bool(kw.name()), None)
            }
            LitKind::Err(val) => (token::Lit::Err(val), None),
        }
    }
}

impl Lit {
    /// Converts literal token with a suffix into an AST literal.
    /// Works speculatively and may return `None` if diagnostic handler is not passed.
    /// If diagnostic handler is passed, may return `Some`,
    /// possibly after reporting non-fatal errors and recovery, or `None` for irrecoverable errors.
    crate fn from_token(
        token: &token::Token,
        span: Span,
        diag: Option<(Span, &Handler)>,
    ) -> Option<Lit> {
        let (token, suffix) = match *token {
            token::Ident(ident, false) if ident.name == keywords::True.name() ||
                                          ident.name == keywords::False.name() =>
                (token::Bool(ident.name), None),
            token::Literal(token, suffix) =>
                (token, suffix),
            token::Interpolated(ref nt) => {
                if let token::NtExpr(expr) | token::NtLiteral(expr) = &**nt {
                    if let ast::ExprKind::Lit(lit) = &expr.node {
                        return Some(lit.clone());
                    }
                }
                return None;
            }
            _ => return None,
        };

        let node = LitKind::from_lit_token(token, suffix, diag)?;
        Some(Lit { node, token, suffix, span })
    }

    /// Attempts to recover an AST literal from semantic literal.
    /// This function is used when the original token doesn't exist (e.g. the literal is created
    /// by an AST-based macro) or unavailable (e.g. from HIR pretty-printing).
    pub fn from_lit_kind(node: LitKind, span: Span) -> Lit {
        let (token, suffix) = node.to_lit_token();
        Lit { node, token, suffix, span }
    }

    /// Losslessly convert an AST literal into a token stream.
    crate fn tokens(&self) -> TokenStream {
        let token = match self.token {
            token::Bool(symbol) => Token::Ident(Ident::with_empty_ctxt(symbol), false),
            token => Token::Literal(token, self.suffix),
        };
        TokenTree::Token(self.span, token).into()
    }
}

impl<'a> Parser<'a> {
    /// Matches `lit = true | false | token_lit`.
    crate fn parse_lit(&mut self) -> PResult<'a, Lit> {
        let diag = Some((self.span, &self.sess.span_diagnostic));
        if let Some(lit) = Lit::from_token(&self.token, self.span, diag) {
            self.bump();
            return Ok(lit);
        } else if self.token == token::Dot {
            // Recover `.4` as `0.4`.
            let recovered = self.look_ahead(1, |t| {
                if let token::Literal(token::Integer(val), suf) = *t {
                    let next_span = self.look_ahead_span(1);
                    if self.span.hi() == next_span.lo() {
                        let sym = String::from("0.") + &val.as_str();
                        let token = token::Literal(token::Float(Symbol::intern(&sym)), suf);
                        return Some((token, self.span.to(next_span)));
                    }
                }
                None
            });
            if let Some((token, span)) = recovered {
                self.diagnostic()
                    .struct_span_err(span, "float literals must have an integer part")
                    .span_suggestion(
                        span,
                        "must have an integer part",
                        pprust::token_to_string(&token),
                        Applicability::MachineApplicable,
                    )
                    .emit();
                let diag = Some((span, &self.sess.span_diagnostic));
                if let Some(lit) = Lit::from_token(&token, span, diag) {
                    self.bump();
                    self.bump();
                    return Ok(lit);
                }
            }
        }

        Err(self.span_fatal(self.span, &format!("unexpected token: {}", self.this_token_descr())))
    }
}

crate fn expect_no_suffix(sp: Span, diag: &Handler, kind: &str, suffix: Option<ast::Name>) {
    match suffix {
        None => {/* everything ok */}
        Some(suf) => {
            let text = suf.as_str();
            if text.is_empty() {
                diag.span_bug(sp, "found empty literal suffix in Some")
            }
            let mut err = if kind == "a tuple index" &&
                ["i32", "u32", "isize", "usize"].contains(&text.to_string().as_str())
            {
                // #59553: warn instead of reject out of hand to allow the fix to percolate
                // through the ecosystem when people fix their macros
                let mut err = diag.struct_span_warn(
                    sp,
                    &format!("suffixes on {} are invalid", kind),
                );
                err.note(&format!(
                    "`{}` is *temporarily* accepted on tuple index fields as it was \
                        incorrectly accepted on stable for a few releases",
                    text,
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
            err.span_label(sp, format!("invalid suffix `{}`", text));
            err.emit();
        }
    }
}

/// Parses a string representing a raw string literal into its final form. The
/// only operation this does is convert embedded CRLF into a single LF.
fn raw_str_lit(lit: &str) -> String {
    debug!("raw_str_lit: given {}", lit.escape_default());
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

// check if `s` looks like i32 or u1234 etc.
fn looks_like_width_suffix(first_chars: &[char], s: &str) -> bool {
    s.starts_with(first_chars) && s[1..].chars().all(|c| c.is_ascii_digit())
}

fn filtered_float_lit(data: Symbol, suffix: Option<Symbol>, diag: Option<(Span, &Handler)>)
                      -> Option<LitKind> {
    debug!("filtered_float_lit: {}, {:?}", data, suffix);
    let suffix = match suffix {
        Some(suffix) => suffix,
        None => return Some(LitKind::FloatUnsuffixed(data)),
    };

    Some(match &*suffix.as_str() {
        "f32" => LitKind::Float(data, ast::FloatTy::F32),
        "f64" => LitKind::Float(data, ast::FloatTy::F64),
        suf => {
            err!(diag, |span, diag| {
                if suf.len() >= 2 && looks_like_width_suffix(&['f'], suf) {
                    // if it looks like a width, lets try to be helpful.
                    let msg = format!("invalid width `{}` for float literal", &suf[1..]);
                    diag.struct_span_err(span, &msg).help("valid widths are 32 and 64").emit()
                } else {
                    let msg = format!("invalid suffix `{}` for float literal", suf);
                    diag.struct_span_err(span, &msg)
                        .span_label(span, format!("invalid suffix `{}`", suf))
                        .help("valid suffixes are `f32` and `f64`")
                        .emit();
                }
            });

            LitKind::FloatUnsuffixed(data)
        }
    })
}
fn float_lit(s: &str, suffix: Option<Symbol>, diag: Option<(Span, &Handler)>)
                 -> Option<LitKind> {
    debug!("float_lit: {:?}, {:?}", s, suffix);
    // FIXME #2252: bounds checking float literals is deferred until trans

    // Strip underscores without allocating a new String unless necessary.
    let s2;
    let s = if s.chars().any(|c| c == '_') {
        s2 = s.chars().filter(|&c| c != '_').collect::<String>();
        &s2
    } else {
        s
    };

    filtered_float_lit(Symbol::intern(s), suffix, diag)
}

fn integer_lit(s: &str, suffix: Option<Symbol>, diag: Option<(Span, &Handler)>)
                   -> Option<LitKind> {
    // s can only be ascii, byte indexing is fine

    // Strip underscores without allocating a new String unless necessary.
    let s2;
    let mut s = if s.chars().any(|c| c == '_') {
        s2 = s.chars().filter(|&c| c != '_').collect::<String>();
        &s2
    } else {
        s
    };

    debug!("integer_lit: {}, {:?}", s, suffix);

    let mut base = 10;
    let orig = s;
    let mut ty = ast::LitIntType::Unsuffixed;

    if s.starts_with('0') && s.len() > 1 {
        match s.as_bytes()[1] {
            b'x' => base = 16,
            b'o' => base = 8,
            b'b' => base = 2,
            _ => { }
        }
    }

    // 1f64 and 2f32 etc. are valid float literals.
    if let Some(suf) = suffix {
        if looks_like_width_suffix(&['f'], &suf.as_str()) {
            let err = match base {
                16 => Some("hexadecimal float literal is not supported"),
                8 => Some("octal float literal is not supported"),
                2 => Some("binary float literal is not supported"),
                _ => None,
            };
            if let Some(err) = err {
                err!(diag, |span, diag| {
                    diag.struct_span_err(span, err)
                        .span_label(span, "not supported")
                        .emit();
                });
            }
            return filtered_float_lit(Symbol::intern(s), Some(suf), diag)
        }
    }

    if base != 10 {
        s = &s[2..];
    }

    if let Some(suf) = suffix {
        if suf.as_str().is_empty() {
            err!(diag, |span, diag| diag.span_bug(span, "found empty literal suffix in Some"));
        }
        ty = match &*suf.as_str() {
            "isize" => ast::LitIntType::Signed(ast::IntTy::Isize),
            "i8"  => ast::LitIntType::Signed(ast::IntTy::I8),
            "i16" => ast::LitIntType::Signed(ast::IntTy::I16),
            "i32" => ast::LitIntType::Signed(ast::IntTy::I32),
            "i64" => ast::LitIntType::Signed(ast::IntTy::I64),
            "i128" => ast::LitIntType::Signed(ast::IntTy::I128),
            "usize" => ast::LitIntType::Unsigned(ast::UintTy::Usize),
            "u8"  => ast::LitIntType::Unsigned(ast::UintTy::U8),
            "u16" => ast::LitIntType::Unsigned(ast::UintTy::U16),
            "u32" => ast::LitIntType::Unsigned(ast::UintTy::U32),
            "u64" => ast::LitIntType::Unsigned(ast::UintTy::U64),
            "u128" => ast::LitIntType::Unsigned(ast::UintTy::U128),
            suf => {
                // i<digits> and u<digits> look like widths, so lets
                // give an error message along those lines
                err!(diag, |span, diag| {
                    if looks_like_width_suffix(&['i', 'u'], suf) {
                        let msg = format!("invalid width `{}` for integer literal", &suf[1..]);
                        diag.struct_span_err(span, &msg)
                            .help("valid widths are 8, 16, 32, 64 and 128")
                            .emit();
                    } else {
                        let msg = format!("invalid suffix `{}` for numeric literal", suf);
                        diag.struct_span_err(span, &msg)
                            .span_label(span, format!("invalid suffix `{}`", suf))
                            .help("the suffix must be one of the integral types \
                                   (`u32`, `isize`, etc)")
                            .emit();
                    }
                });

                ty
            }
        }
    }

    debug!("integer_lit: the type is {:?}, base {:?}, the new string is {:?}, the original \
           string was {:?}, the original suffix was {:?}", ty, base, s, orig, suffix);

    Some(match u128::from_str_radix(s, base) {
        Ok(r) => LitKind::Int(r, ty),
        Err(_) => {
            // small bases are lexed as if they were base 10, e.g, the string
            // might be `0b10201`. This will cause the conversion above to fail,
            // but these cases have errors in the lexer: we don't want to emit
            // two errors, and we especially don't want to emit this error since
            // it isn't necessarily true.
            let already_errored = base < 10 &&
                s.chars().any(|c| c.to_digit(10).map_or(false, |d| d >= base));

            if !already_errored {
                err!(diag, |span, diag| diag.span_err(span, "int literal is too large"));
            }
            LitKind::Int(0, ty)
        }
    })
}
