//! proc-macro server implementation
//!
//! Based on idea from <https://github.com/fedochet/rust-proc-macro-expander>
//! The lib-proc-macro server backend is `TokenStream`-agnostic, such that
//! we could provide any TokenStream implementation.
//! The original idea from fedochet is using proc-macro2 as backend,
//! we use tt instead for better integration with RA.
//!
//! FIXME: No span and source file information is implemented yet

use proc_macro::bridge;

mod token_stream;
pub use token_stream::TokenStream;

pub mod rust_analyzer_span;
mod symbol;
pub mod token_id;
pub use symbol::*;
use tt::Spacing;

fn delim_to_internal<S>(d: proc_macro::Delimiter, span: bridge::DelimSpan<S>) -> tt::Delimiter<S> {
    let kind = match d {
        proc_macro::Delimiter::Parenthesis => tt::DelimiterKind::Parenthesis,
        proc_macro::Delimiter::Brace => tt::DelimiterKind::Brace,
        proc_macro::Delimiter::Bracket => tt::DelimiterKind::Bracket,
        proc_macro::Delimiter::None => tt::DelimiterKind::Invisible,
    };
    tt::Delimiter { open: span.open, close: span.close, kind }
}

fn delim_to_external<S>(d: tt::Delimiter<S>) -> proc_macro::Delimiter {
    match d.kind {
        tt::DelimiterKind::Parenthesis => proc_macro::Delimiter::Parenthesis,
        tt::DelimiterKind::Brace => proc_macro::Delimiter::Brace,
        tt::DelimiterKind::Bracket => proc_macro::Delimiter::Bracket,
        tt::DelimiterKind::Invisible => proc_macro::Delimiter::None,
    }
}

#[allow(unused)]
fn spacing_to_internal(spacing: proc_macro::Spacing) -> Spacing {
    match spacing {
        proc_macro::Spacing::Alone => Spacing::Alone,
        proc_macro::Spacing::Joint => Spacing::Joint,
    }
}

#[allow(unused)]
fn spacing_to_external(spacing: Spacing) -> proc_macro::Spacing {
    match spacing {
        Spacing::Alone => proc_macro::Spacing::Alone,
        Spacing::Joint => proc_macro::Spacing::Joint,
    }
}

/// Invokes the callback with a `&[&str]` consisting of each part of the
/// literal's representation. This is done to allow the `ToString` and
/// `Display` implementations to borrow references to symbol values, and
/// both be optimized to reduce overhead.
fn literal_with_stringify_parts<S, R>(
    literal: &bridge::Literal<S, Symbol>,
    interner: SymbolInternerRef,
    f: impl FnOnce(&[&str]) -> R,
) -> R {
    /// Returns a string containing exactly `num` '#' characters.
    /// Uses a 256-character source string literal which is always safe to
    /// index with a `u8` index.
    fn get_hashes_str(num: u8) -> &'static str {
        const HASHES: &str = "\
                        ################################################################\
                        ################################################################\
                        ################################################################\
                        ################################################################\
                        ";
        const _: () = assert!(HASHES.len() == 256);
        &HASHES[..num as usize]
    }

    {
        let symbol = &*literal.symbol.text(interner);
        let suffix = &*literal.suffix.map(|s| s.text(interner)).unwrap_or_default();
        match literal.kind {
            bridge::LitKind::Byte => f(&["b'", symbol, "'", suffix]),
            bridge::LitKind::Char => f(&["'", symbol, "'", suffix]),
            bridge::LitKind::Str => f(&["\"", symbol, "\"", suffix]),
            bridge::LitKind::StrRaw(n) => {
                let hashes = get_hashes_str(n);
                f(&["r", hashes, "\"", symbol, "\"", hashes, suffix])
            }
            bridge::LitKind::ByteStr => f(&["b\"", symbol, "\"", suffix]),
            bridge::LitKind::ByteStrRaw(n) => {
                let hashes = get_hashes_str(n);
                f(&["br", hashes, "\"", symbol, "\"", hashes, suffix])
            }
            bridge::LitKind::CStr => f(&["c\"", symbol, "\"", suffix]),
            bridge::LitKind::CStrRaw(n) => {
                let hashes = get_hashes_str(n);
                f(&["cr", hashes, "\"", symbol, "\"", hashes, suffix])
            }
            bridge::LitKind::Integer | bridge::LitKind::Float | bridge::LitKind::ErrWithGuar => {
                f(&[symbol, suffix])
            }
        }
    }
}
