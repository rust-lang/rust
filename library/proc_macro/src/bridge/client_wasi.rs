//! This module implements the client (in-wasm) for WebAssembly-based proc macros.
//!
//! At a high level, unlike for native proc macros, we go through a component model interface that
//! mirrors the API defined in bridge/mod.rs (see library/proc_macro/wasm-interface.wit). Where
//! possible, we directly use types defined by WIT in the bridge interface, but where needed we have
//! wrappers around it. The raw interface is mandated to be closely mirrored because we use
//! `with_api!` to generate it (see bottom of this file). From there input/output is converted with
//! the `Encode` (wasm -> host) and `Decode` (host -> wasm) traits. These are similar to those used
//! for the buffer encoding in native code, but typically just map types with very little real
//! logic (unlike the buffer encoding), low-level serde is handled by wit-bindgen.

use std::fmt;
use std::ops::Bound;

pub(crate) use host::TokenStream;

pub(crate) use crate::bridge::Methods;
pub(crate) use crate::bridge::symbol::Symbol;
use crate::wasi_bindgen::exports::rust_lang::rust::custom_derive as cd;
use crate::wasi_bindgen::rust_lang::rust::host;

pub(crate) fn is_available() -> bool {
    // On wasm we currently don't support executing libproc_macro without the bridge (since it's
    // statically imported) -- unclear whether that can change in the future, but for now just
    // always return true.
    true
}

#[derive(Copy, Clone)]
pub(crate) struct Span(pub(crate) host::Span);

impl Span {
    pub(crate) fn def_site() -> Span {
        Span(host::span_def_site())
    }

    pub(crate) fn call_site() -> Span {
        Span(host::span_call_site())
    }

    pub(crate) fn mixed_site() -> Span {
        Span(host::span_mixed_site())
    }

    pub(crate) fn byte_range(self) -> std::ops::Range<usize> {
        Methods::span_byte_range(self)
    }

    pub(crate) fn parent(self) -> Option<Span> {
        Methods::span_parent(self)
    }

    pub(crate) fn source(self) -> Span {
        Methods::span_source(self)
    }
}

impl PartialEq for Span {
    fn eq(&self, other: &Span) -> bool {
        host::span_is_same(self.0, other.0)
    }
}

impl fmt::Debug for Span {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&host::span_debug(self.0))
    }
}

impl Clone for TokenStream {
    fn clone(&self) -> Self {
        host::ts_clone(self)
    }
}

impl !Send for TokenStream {}
impl !Sync for TokenStream {}

#[derive(Copy, Clone)]
pub enum Client {
    Expand1 { expand: fn(crate::TokenStream) -> crate::TokenStream },
    Expand2 { expand: fn(crate::TokenStream, crate::TokenStream) -> crate::TokenStream },
}

impl Client {
    pub const fn expand1(expand: fn(crate::TokenStream) -> crate::TokenStream) -> Self {
        Self::Expand1 { expand }
    }

    pub const fn expand2(
        expand: fn(crate::TokenStream, crate::TokenStream) -> crate::TokenStream,
    ) -> Self {
        Self::Expand2 { expand }
    }
}

impl cd::GuestCustomDerive for Client {
    fn get_kind(&self) -> cd::DeriveKind {
        match self {
            Self::Expand1 { .. } => cd::DeriveKind::Expand1,
            Self::Expand2 { .. } => cd::DeriveKind::Expand2,
        }
    }

    fn expand1(&self, tokens: TokenStream) -> TokenStream {
        setup_hook();
        match self {
            Self::Expand1 { expand } => {
                let output = (*expand)(crate::TokenStream(Some(tokens)));
                let output = output.0.unwrap_or_else(|| TokenStream::new());
                output
            }
            Self::Expand2 { .. } => unreachable!("should not be called by host"),
        }
    }

    fn expand2(&self, a: TokenStream, b: TokenStream) -> TokenStream {
        setup_hook();

        match self {
            Self::Expand2 { expand } => {
                let output = (*expand)(crate::TokenStream(Some(a)), crate::TokenStream(Some(b)));
                let output = output.0.unwrap_or_else(|| TokenStream::new());
                output
            }
            Self::Expand1 { .. } => {
                unreachable!("should not be called by host")
            }
        }
    }
}

fn setup_hook() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let prev = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            // If the host wants us to also emit it, run the default panic hook.
            if host::report_panic(info.payload_as_str()) {
                prev(info)
            }
        }));
    });
}

#[allow_internal_unstable(proc_macro_internals, staged_api)]
pub macro generate_export($list:ident) {
    use $crate::wasi_bindgen::exports::rust_lang::rust::custom_derive as cd;

    struct ExportAbi;

    impl cd::Guest for ExportAbi {
        type CustomDerive = $crate::bridge::client::Client;
        fn get_custom_derives() -> Vec<cd::CustomDerive> {
            $list.iter().map(|d| cd::CustomDerive::new(*d)).collect()
        }
    }

    $crate::wasi_bindgen::export!(ExportAbi with_types_in $crate::wasi_bindgen);
}

trait Encode<Output> {
    fn convert(input: Self) -> Output;
}

impl Decode<host::LiteralKind> for super::LitKind {
    fn convert(v: super::LitKind) -> host::LiteralKind {
        match v {
            super::LitKind::Byte => host::LiteralKind::Byte,
            super::LitKind::Char => host::LiteralKind::Char,
            super::LitKind::Integer => host::LiteralKind::Integer,
            super::LitKind::Float => host::LiteralKind::Float,
            super::LitKind::ByteStr => host::LiteralKind::ByteStr,
            super::LitKind::ByteStrRaw(n) => host::LiteralKind::ByteStrRaw(n),
            super::LitKind::Str => host::LiteralKind::Str,
            super::LitKind::StrRaw(n) => host::LiteralKind::StrRaw(n),
            super::LitKind::CStr => host::LiteralKind::CStr,
            super::LitKind::CStrRaw(n) => host::LiteralKind::CStrRaw(n),
            super::LitKind::ErrWithGuar => host::LiteralKind::ErrWithGuar,
        }
    }
}

impl Encode<super::LitKind> for host::LiteralKind {
    fn convert(v: Self) -> super::LitKind {
        match v {
            host::LiteralKind::Byte => super::LitKind::Byte,
            host::LiteralKind::Char => super::LitKind::Char,
            host::LiteralKind::Integer => super::LitKind::Integer,
            host::LiteralKind::Float => super::LitKind::Float,
            host::LiteralKind::ByteStr => super::LitKind::ByteStr,
            host::LiteralKind::ByteStrRaw(n) => super::LitKind::ByteStrRaw(n),
            host::LiteralKind::Str => super::LitKind::Str,
            host::LiteralKind::StrRaw(n) => super::LitKind::StrRaw(n),
            host::LiteralKind::CStr => super::LitKind::CStr,
            host::LiteralKind::CStrRaw(n) => super::LitKind::CStrRaw(n),
            host::LiteralKind::ErrWithGuar => super::LitKind::ErrWithGuar,
        }
    }
}

impl Encode<Result<Symbol, ()>> for Result<String, ()> {
    fn convert(input: Self) -> Result<Symbol, ()> {
        match input {
            Ok(v) => Ok(Symbol::new(&v)),
            Err(()) => Err(()),
        }
    }
}

impl Encode<String> for String {
    fn convert(input: Self) -> String {
        input
    }
}

impl Encode<Span> for host::Span {
    fn convert(input: Self) -> Span {
        Span(input)
    }
}

impl Encode<Option<Span>> for Option<host::Span> {
    fn convert(input: Self) -> Option<Span> {
        input.map(Span)
    }
}

impl Encode<usize> for u64 {
    fn convert(input: Self) -> usize {
        input.try_into().unwrap()
    }
}

trait Decode<Out> {
    fn convert(input: Self) -> Out;
}

impl<'a> Decode<&'a str> for &'a str {
    fn convert(input: &str) -> &str {
        input
    }
}

impl Decode<u64> for usize {
    fn convert(input: Self) -> u64 {
        input.try_into().unwrap()
    }
}

impl Decode<host::Span> for Span {
    fn convert(input: Self) -> host::Span {
        input.0
    }
}

impl Decode<host::Diagnostic> for crate::bridge::Diagnostic<Span> {
    fn convert(input: Self) -> host::Diagnostic {
        fn to_internal(diag: crate::bridge::Diagnostic<Span>) -> host::Diagnostic {
            host::Diagnostic::new(host::DiagnosticInner {
                level: match diag.level {
                    crate::Level::Error => host::Level::Error,
                    crate::Level::Warning => host::Level::Warning,
                    crate::Level::Note => host::Level::Note,
                    crate::Level::Help => host::Level::Help,
                },
                message: diag.message,
                spans: diag.spans.into_iter().map(|s| s.0).collect(),
                children: diag.children.into_iter().map(to_internal).collect(),
            })
        }

        to_internal(input)
    }
}

impl Encode<Vec<super::TokenTree<TokenStream, Span, Symbol>>> for Vec<host::TokenTree> {
    fn convert(input: Self) -> Vec<super::TokenTree<TokenStream, Span, Symbol>> {
        input.into_iter().map(Encode::convert).collect()
    }
}

impl Encode<super::TokenTree<TokenStream, Span, Symbol>> for host::TokenTree {
    fn convert(input: Self) -> super::TokenTree<TokenStream, Span, Symbol> {
        match input {
            host::TokenTree::Group(g) => super::TokenTree::Group(super::Group {
                delimiter: match g.delimiter {
                    host::Delimiter::Parenthesis => super::Delimiter::Parenthesis,
                    host::Delimiter::Brace => super::Delimiter::Brace,
                    host::Delimiter::Bracket => super::Delimiter::Bracket,
                    host::Delimiter::None => super::Delimiter::None,
                },
                stream: g.stream,
                span: super::DelimSpan {
                    open: Span(g.span.open),
                    close: Span(g.span.close),
                    entire: Span(g.span.entire),
                },
            }),
            host::TokenTree::Punct(p) => super::TokenTree::Punct(super::Punct {
                ch: p.ch,
                joint: p.joint,
                span: Span(p.span),
            }),
            host::TokenTree::Ident(i) => super::TokenTree::Ident(super::Ident {
                sym: Symbol::new(&i.sym),
                is_raw: i.is_raw,
                span: Span(i.span),
            }),
            host::TokenTree::Literal(l) => super::TokenTree::Literal(super::Literal {
                kind: Encode::convert(l.kind),
                span: Span(l.span),
                suffix: l.suffix.map(|s| Symbol::new(&s)),
                symbol: Symbol::new(&l.symbol),
            }),
        }
    }
}

impl Decode<Vec<TokenStream>> for Vec<TokenStream> {
    fn convert(input: Self) -> Vec<TokenStream> {
        input.into_iter().map(Decode::convert).collect()
    }
}

impl Decode<host::RangeBound> for Bound<usize> {
    fn convert(input: Self) -> host::RangeBound {
        match input {
            Bound::Included(v) => {
                host::RangeBound { value: v as u64, bound: host::Bound::Included }
            }
            Bound::Excluded(v) => {
                host::RangeBound { value: v as u64, bound: host::Bound::Excluded }
            }
            Bound::Unbounded => host::RangeBound { value: 0, bound: host::Bound::Unbounded },
        }
    }
}

impl Encode<std::ops::Range<usize>> for (u64, u64) {
    fn convert(input: Self) -> std::ops::Range<usize> {
        input.0.try_into().unwrap()..input.1.try_into().unwrap()
    }
}

impl Encode<Option<String>> for Option<String> {
    fn convert(input: Self) -> Option<String> {
        input
    }
}

impl<'a> Decode<Option<&'a str>> for Option<&'a str> {
    fn convert(input: Option<&str>) -> Option<&str> {
        input
    }
}

impl Encode<()> for () {
    fn convert(_: ()) -> () {}
}

impl Encode<bool> for bool {
    fn convert(v: bool) -> bool {
        v
    }
}

impl Encode<Result<super::Literal<Span, Symbol>, String>> for Result<host::Literal, String> {
    fn convert(input: Self) -> Result<super::Literal<Span, Symbol>, String> {
        match input {
            Ok(l) => Ok(super::Literal {
                kind: Encode::convert(l.kind),
                symbol: Symbol::new(&l.symbol),
                suffix: l.suffix.map(|s| Symbol::new(&s)),
                span: Span(l.span),
            }),
            Err(e) => Err(e),
        }
    }
}

impl Decode<TokenStream> for TokenStream {
    fn convert(input: TokenStream) -> TokenStream {
        input
    }
}

impl<'a> Decode<&'a TokenStream> for &'a TokenStream {
    fn convert(input: &TokenStream) -> &TokenStream {
        &input
    }
}

impl Encode<TokenStream> for TokenStream {
    fn convert(input: TokenStream) -> TokenStream {
        input
    }
}

impl Encode<Result<TokenStream, ()>> for Result<TokenStream, ()> {
    fn convert(input: Result<TokenStream, ()>) -> Result<TokenStream, ()> {
        input
    }
}

impl Encode<Result<TokenStream, String>> for Result<TokenStream, String> {
    fn convert(input: Result<TokenStream, String>) -> Result<TokenStream, String> {
        input
    }
}

impl Decode<host::TokenTree> for super::TokenTree<TokenStream, Span, Symbol> {
    fn convert(input: Self) -> host::TokenTree {
        match input {
            super::TokenTree::Group(g) => host::TokenTree::Group(host::Group {
                delimiter: match g.delimiter {
                    super::Delimiter::Parenthesis => host::Delimiter::Parenthesis,
                    super::Delimiter::Brace => host::Delimiter::Brace,
                    super::Delimiter::Bracket => host::Delimiter::Bracket,
                    super::Delimiter::None => host::Delimiter::None,
                },
                stream: g.stream,
                span: host::DelimSpan {
                    open: g.span.open.0,
                    close: g.span.close.0,
                    entire: g.span.entire.0,
                },
            }),
            super::TokenTree::Punct(p) => {
                host::TokenTree::Punct(host::Punct { ch: p.ch, joint: p.joint, span: p.span.0 })
            }
            super::TokenTree::Ident(i) => host::TokenTree::Ident(host::Ident {
                sym: i.sym.to_string(),
                is_raw: i.is_raw,
                span: i.span.0,
            }),
            super::TokenTree::Literal(l) => host::TokenTree::Literal(host::Literal {
                kind: Decode::convert(l.kind),
                span: l.span.0,
                suffix: l.suffix.map(|s| s.to_string()),
                symbol: l.symbol.to_string(),
            }),
        }
    }
}

impl Decode<Option<TokenStream>> for Option<TokenStream> {
    fn convert(input: Self) -> Option<TokenStream> {
        input
    }
}

impl Decode<Vec<host::TokenTree>> for Vec<super::TokenTree<TokenStream, Span, Symbol>> {
    fn convert(input: Self) -> Vec<host::TokenTree> {
        input.into_iter().map(Decode::convert).collect()
    }
}

macro_rules! wasm {
    (
        $(fn $method:ident($($arg:ident: $arg_ty:ty),* $(,)?) $(-> $ret_ty:ty)?;)*
    ) => {
        impl Methods {
            $(
                pub(crate) fn $method($($arg: $arg_ty),*) $(-> $ret_ty)* {
                    Encode::convert(host::$method($(Decode::convert($arg),)+))
                }
            )*
        }
    }
}

mod api_instantiated {
    use std::ops::{Bound, Range};

    use super::{Decode, Encode, Span, TokenStream, host};
    use crate::bridge::symbol::Symbol;
    use crate::bridge::{Diagnostic, Literal, Methods, TokenTree};

    with_api!(wasm, TokenStream, Span, Symbol);
}
