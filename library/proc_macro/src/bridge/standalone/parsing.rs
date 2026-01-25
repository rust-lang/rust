use super::{Literal, Result, Span};
use crate::bridge::LitKind;
use crate::bridge::client::Symbol;
use crate::get_hashes_str;

fn parse_maybe_raw_str(
    mut s: &str,
    raw_variant: fn(u8) -> LitKind,
    regular_variant: LitKind,
) -> Result<Literal> {
    let mut hash_count = None;

    if s.starts_with('r') {
        s = s.strip_prefix('r').unwrap();
        let mut h = 0;
        for c in s.chars() {
            if c == '#' {
                if h == u8::MAX {
                    return Err(());
                }
                h += 1;
            } else {
                break;
            }
        }
        hash_count = Some(h);
        let hashes = get_hashes_str(h);
        s = s.strip_prefix(hashes).unwrap();
        s = s.strip_suffix(hashes).ok_or(())?;
    }
    let sym = parse_plain_str(s)?;

    Ok(make_literal(if let Some(h) = hash_count { raw_variant(h) } else { regular_variant }, sym))
}

fn parse_char(s: &str) -> Result<Literal> {
    if let Some(s) = s.strip_circumfix('\'', '\'') {
        if s.chars().count() == 1 {
            Ok(make_literal(LitKind::Char, Symbol::new(s)))
        } else {
            Err(())
        }
    } else {
        Err(())
    }
}

fn parse_plain_str(s: &str) -> Result<Symbol> {
    Ok(Symbol::new(s.strip_circumfix("\"", "\"").ok_or(())?))
}

const INT_SUFFIXES: &[&str] =
    &["u8", "i8", "u16", "i16", "u32", "i32", "u64", "i64", "u128", "i128"];
const FLOAT_SUFFIXES: &[&str] = &["f16", "f32", "f64", "f128"];

fn parse_numeral(s: &str) -> Result<Literal> {
    for suffix in INT_SUFFIXES {
        if s.ends_with(suffix) {
            return parse_integer(s);
        }
    }
    let non_negative = s.trim_prefix('-');
    if non_negative.starts_with("0b")
        || non_negative.starts_with("0o")
        || non_negative.starts_with("0x")
    {
        return parse_integer(s);
    }
    for suffix in FLOAT_SUFFIXES {
        if s.ends_with(suffix) {
            return parse_float(s);
        }
    }
    if s.contains('.') || s.contains('e') || s.contains('E') {
        return parse_float(s);
    }

    parse_integer(s)
}

fn parse_float(symbol: &str) -> Result<Literal> {
    let (s, suffix) = strip_number_suffix(symbol, FLOAT_SUFFIXES);
    Ok(Literal { kind: LitKind::Float, symbol: Symbol::new(s), suffix, span: Span })
}

fn parse_integer(symbol: &str) -> Result<Literal> {
    let (symbol, suffix) = strip_number_suffix(symbol, INT_SUFFIXES);
    let s = symbol.trim_prefix('-');
    let (s, valid_chars) = if let Some(s) = s.strip_prefix("0b") {
        (s, ('0'..='1').collect::<Vec<_>>())
    } else if let Some(s) = s.strip_prefix("0o") {
        (s, ('0'..='7').collect())
    } else if let Some(s) = s.strip_prefix("0x") {
        (s, ('0'..='9').chain('a'..='f').chain('A'..='F').collect())
    } else {
        (s, ('0'..='9').collect())
    };

    let mut any_found = false;
    for c in s.chars() {
        if c == '_' {
            continue;
        }
        if valid_chars.contains(&c) {
            any_found = true;
            continue;
        }
        return Err(());
    }
    if !any_found {
        return Err(());
    }

    Ok(Literal { kind: LitKind::Integer, symbol: Symbol::new(symbol), suffix, span: Span })
}

fn strip_number_suffix<'a>(s: &'a str, suffixes: &[&str]) -> (&'a str, Option<Symbol>) {
    for suf in suffixes {
        if let Some(new_s) = s.strip_suffix(suf) {
            return (new_s, Some(Symbol::new(suf)));
        }
    }
    (s, None)
}

fn make_literal(kind: LitKind, symbol: Symbol) -> Literal {
    Literal { kind, symbol, suffix: None, span: Span }
}

pub(super) fn literal_from_str(s: &str) -> Result<Literal> {
    let s = &crate::escape::escape_bytes(
        s.as_bytes(),
        crate::escape::EscapeOptions {
            escape_single_quote: false,
            escape_double_quote: false,
            escape_nonascii: false,
        },
    );
    let mut chars = s.chars();
    let first = chars.next().ok_or(())?;
    let rest = &s[1..];
    match first {
        'b' => {
            if chars.next() == Some('\'') {
                parse_char(rest).map(|mut lit| {
                    lit.kind = LitKind::Byte;
                    lit
                })
            } else {
                parse_maybe_raw_str(rest, LitKind::ByteStrRaw, LitKind::ByteStr)
            }
        }
        'c' => parse_maybe_raw_str(rest, LitKind::CStrRaw, LitKind::CStr),
        'r' => parse_maybe_raw_str(s, LitKind::StrRaw, LitKind::Str),
        '0'..='9' | '-' => parse_numeral(s),
        '\'' => parse_char(s),
        '"' => Ok(make_literal(LitKind::Str, parse_plain_str(s)?)),
        _ => Err(()),
    }
}
