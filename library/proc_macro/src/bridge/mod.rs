//! Internal interface for communicating between a `proc_macro` client
//! (a proc macro crate) and a `proc_macro` server (a compiler front-end).
//!
//! Serialization (with C ABI buffers) and unique integer handles are employed
//! to allow safely interfacing between two copies of `proc_macro` built
//! (from the same source) by different compilers with potentially mismatching
//! Rust ABIs (e.g., stage0/bin/rustc vs stage1/bin/rustc during bootstrap).

#![deny(unsafe_code)]

use std::hash::Hash;
use std::ops::{Bound, Range};
use std::sync::Once;
use std::{fmt, marker, mem, panic, thread};

use crate::{Delimiter, Level};

/// Higher-order macro describing the server RPC API, allowing automatic
/// generation of type-safe Rust APIs, both client-side and server-side.
///
/// `with_api!(MySelf, my_macro)` expands to:
/// ```rust,ignore (pseudo-code)
/// my_macro! {
///     fn lit_character(ch: char) -> MySelf::Literal;
///     fn lit_span(lit: &MySelf::Literal) -> MySelf::Span;
///     fn lit_set_span(lit: &mut MySelf::Literal, span: MySelf::Span);
///     // ...
/// }
/// ```
///
/// The first argument serves to customize the argument/return types,
/// to enable several different usecases:
///
/// If `MySelf` is just `Self`, then the types are only valid inside
/// a trait or a trait impl, where the trait has associated types
/// for each of the API types. If non-associated types are desired,
/// a module name (`self` in practice) can be used instead of `Self`.
macro_rules! with_api {
    ($S:ident, $m:ident) => {
        $m! {
            fn injected_env_var(var: &str) -> Option<String>;
            fn track_env_var(var: &str, value: Option<&str>);
            fn track_path(path: &str);
            fn literal_from_str(s: &str) -> Result<Literal<$S::Span, $S::Symbol>, ()>;
            fn emit_diagnostic(diagnostic: Diagnostic<$S::Span>);

            fn ts_drop(stream: $S::TokenStream);
            fn ts_clone(stream: &$S::TokenStream) -> $S::TokenStream;
            fn ts_is_empty(stream: &$S::TokenStream) -> bool;
            fn ts_expand_expr(stream: &$S::TokenStream) -> Result<$S::TokenStream, ()>;
            fn ts_from_str(src: &str) -> $S::TokenStream;
            fn ts_to_string(stream: &$S::TokenStream) -> String;
            fn ts_from_token_tree(
                tree: TokenTree<$S::TokenStream, $S::Span, $S::Symbol>,
            ) -> $S::TokenStream;
            fn ts_concat_trees(
                base: Option<$S::TokenStream>,
                trees: Vec<TokenTree<$S::TokenStream, $S::Span, $S::Symbol>>,
            ) -> $S::TokenStream;
            fn ts_concat_streams(
                base: Option<$S::TokenStream>,
                streams: Vec<$S::TokenStream>,
            ) -> $S::TokenStream;
            fn ts_into_trees(
                stream: $S::TokenStream
            ) -> Vec<TokenTree<$S::TokenStream, $S::Span, $S::Symbol>>;

            fn span_debug(span: $S::Span) -> String;
            fn span_parent(span: $S::Span) -> Option<$S::Span>;
            fn span_source(span: $S::Span) -> $S::Span;
            fn span_byte_range(span: $S::Span) -> Range<usize>;
            fn span_start(span: $S::Span) -> $S::Span;
            fn span_end(span: $S::Span) -> $S::Span;
            fn span_line(span: $S::Span) -> usize;
            fn span_column(span: $S::Span) -> usize;
            fn span_file(span: $S::Span) -> String;
            fn span_local_file(span: $S::Span) -> Option<String>;
            fn span_join(span: $S::Span, other: $S::Span) -> Option<$S::Span>;
            fn span_subspan(span: $S::Span, start: Bound<usize>, end: Bound<usize>) -> Option<$S::Span>;
            fn span_resolved_at(span: $S::Span, at: $S::Span) -> $S::Span;
            fn span_source_text(span: $S::Span) -> Option<String>;
            fn span_save_span(span: $S::Span) -> usize;
            fn span_recover_proc_macro_span(id: usize) -> $S::Span;

            fn symbol_normalize_and_validate_ident(string: &str) -> Result<$S::Symbol, ()>;
        }
    };
}

pub(crate) struct Methods;

#[allow(unsafe_code)]
mod arena;
#[allow(unsafe_code)]
mod buffer;
#[deny(unsafe_code)]
pub mod client;
#[allow(unsafe_code)]
mod closure;
#[forbid(unsafe_code)]
mod fxhash;
#[forbid(unsafe_code)]
mod handle;
#[macro_use]
#[forbid(unsafe_code)]
mod rpc;
#[allow(unsafe_code)]
mod selfless_reify;
#[forbid(unsafe_code)]
pub mod server;
#[allow(unsafe_code)]
mod symbol;

use buffer::Buffer;
pub use rpc::PanicMessage;
use rpc::{Decode, Encode};

/// Configuration for establishing an active connection between a server and a
/// client.  The server creates the bridge config (`run_server` in `server.rs`),
/// then passes it to the client through the function pointer in the `run` field
/// of `client::Client`. The client constructs a local `Bridge` from the config
/// in TLS during its execution (`Bridge::{enter, with}` in `client.rs`).
#[repr(C)]
pub struct BridgeConfig<'a> {
    /// Buffer used to pass initial input to the client.
    input: Buffer,

    /// Server-side function that the client uses to make requests.
    dispatch: closure::Closure<'a, Buffer, Buffer>,

    /// If 'true', always invoke the default panic hook
    force_show_panics: bool,
}

impl !Send for BridgeConfig<'_> {}
impl !Sync for BridgeConfig<'_> {}

macro_rules! declare_tags {
    (
        $(fn $method:ident($($arg:ident: $arg_ty:ty),* $(,)?) $(-> $ret_ty:ty)*;)*
    ) => {
        #[allow(non_camel_case_types)]
        pub(super) enum ApiTags {
            $($method),*
        }
        rpc_encode_decode!(enum ApiTags { $($method),* });
    }
}
with_api!(self, declare_tags);

/// Helper to wrap associated types to allow trait impl dispatch.
/// That is, normally a pair of impls for `T::Foo` and `T::Bar`
/// can overlap, but if the impls are, instead, on types like
/// `Marked<T::Foo, Foo>` and `Marked<T::Bar, Bar>`, they can't.
trait Mark {
    type Unmarked;
    fn mark(unmarked: Self::Unmarked) -> Self;
    fn unmark(self) -> Self::Unmarked;
}

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
struct Marked<T, M> {
    value: T,
    _marker: marker::PhantomData<M>,
}

impl<T, M> Mark for Marked<T, M> {
    type Unmarked = T;
    fn mark(unmarked: Self::Unmarked) -> Self {
        Marked { value: unmarked, _marker: marker::PhantomData }
    }
    fn unmark(self) -> Self::Unmarked {
        self.value
    }
}
impl<'a, T, M> Mark for &'a Marked<T, M> {
    type Unmarked = &'a T;
    fn mark(_: Self::Unmarked) -> Self {
        unreachable!()
    }
    fn unmark(self) -> Self::Unmarked {
        &self.value
    }
}

impl<T: Mark> Mark for Vec<T> {
    type Unmarked = Vec<T::Unmarked>;
    fn mark(unmarked: Self::Unmarked) -> Self {
        // Should be a no-op due to std's in-place collect optimizations.
        unmarked.into_iter().map(T::mark).collect()
    }
    fn unmark(self) -> Self::Unmarked {
        // Should be a no-op due to std's in-place collect optimizations.
        self.into_iter().map(T::unmark).collect()
    }
}

macro_rules! mark_noop {
    ($($ty:ty),* $(,)?) => {
        $(
            impl Mark for $ty {
                type Unmarked = Self;
                fn mark(unmarked: Self::Unmarked) -> Self {
                    unmarked
                }
                fn unmark(self) -> Self::Unmarked {
                    self
                }
            }
        )*
    }
}
mark_noop! {
    (),
    bool,
    &'_ str,
    String,
    u8,
    usize,
    Delimiter,
    LitKind,
    Level,
}

rpc_encode_decode!(
    enum Delimiter {
        Parenthesis,
        Brace,
        Bracket,
        None,
    }
);
rpc_encode_decode!(
    enum Level {
        Error,
        Warning,
        Note,
        Help,
    }
);

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum LitKind {
    Byte,
    Char,
    Integer,
    Float,
    Str,
    StrRaw(u8),
    ByteStr,
    ByteStrRaw(u8),
    CStr,
    CStrRaw(u8),
    // This should have an `ErrorGuaranteed`, except that type isn't available
    // in this crate. (Imagine it is there.) Hence the `WithGuar` suffix. Must
    // only be constructed in `LitKind::from_internal`, where an
    // `ErrorGuaranteed` is available.
    ErrWithGuar,
}

rpc_encode_decode!(
    enum LitKind {
        Byte,
        Char,
        Integer,
        Float,
        Str,
        StrRaw(n),
        ByteStr,
        ByteStrRaw(n),
        CStr,
        CStrRaw(n),
        ErrWithGuar,
    }
);

macro_rules! mark_compound {
    (struct $name:ident <$($T:ident),+> { $($field:ident),* $(,)? }) => {
        impl<$($T: Mark),+> Mark for $name <$($T),+> {
            type Unmarked = $name <$($T::Unmarked),+>;
            fn mark(unmarked: Self::Unmarked) -> Self {
                $name {
                    $($field: Mark::mark(unmarked.$field)),*
                }
            }
            fn unmark(self) -> Self::Unmarked {
                $name {
                    $($field: Mark::unmark(self.$field)),*
                }
            }
        }
    };
    (enum $name:ident <$($T:ident),+> { $($variant:ident $(($field:ident))?),* $(,)? }) => {
        impl<$($T: Mark),+> Mark for $name <$($T),+> {
            type Unmarked = $name <$($T::Unmarked),+>;
            fn mark(unmarked: Self::Unmarked) -> Self {
                match unmarked {
                    $($name::$variant $(($field))? => {
                        $name::$variant $((Mark::mark($field)))?
                    })*
                }
            }
            fn unmark(self) -> Self::Unmarked {
                match self {
                    $($name::$variant $(($field))? => {
                        $name::$variant $((Mark::unmark($field)))?
                    })*
                }
            }
        }
    }
}

macro_rules! compound_traits {
    ($($t:tt)*) => {
        rpc_encode_decode!($($t)*);
        mark_compound!($($t)*);
    };
}

compound_traits!(
    enum Bound<T> {
        Included(x),
        Excluded(x),
        Unbounded,
    }
);

compound_traits!(
    enum Option<T> {
        Some(t),
        None,
    }
);

compound_traits!(
    enum Result<T, E> {
        Ok(t),
        Err(e),
    }
);

#[derive(Copy, Clone)]
pub struct DelimSpan<Span> {
    pub open: Span,
    pub close: Span,
    pub entire: Span,
}

impl<Span: Copy> DelimSpan<Span> {
    pub fn from_single(span: Span) -> Self {
        DelimSpan { open: span, close: span, entire: span }
    }
}

compound_traits!(struct DelimSpan<Span> { open, close, entire });

#[derive(Clone)]
pub struct Group<TokenStream, Span> {
    pub delimiter: Delimiter,
    pub stream: Option<TokenStream>,
    pub span: DelimSpan<Span>,
}

compound_traits!(struct Group<TokenStream, Span> { delimiter, stream, span });

#[derive(Clone)]
pub struct Punct<Span> {
    pub ch: u8,
    pub joint: bool,
    pub span: Span,
}

compound_traits!(struct Punct<Span> { ch, joint, span });

#[derive(Copy, Clone, Eq, PartialEq)]
pub struct Ident<Span, Symbol> {
    pub sym: Symbol,
    pub is_raw: bool,
    pub span: Span,
}

compound_traits!(struct Ident<Span, Symbol> { sym, is_raw, span });

#[derive(Clone, Eq, PartialEq)]
pub struct Literal<Span, Symbol> {
    pub kind: LitKind,
    pub symbol: Symbol,
    pub suffix: Option<Symbol>,
    pub span: Span,
}

compound_traits!(struct Literal<Sp, Sy> { kind, symbol, suffix, span });

#[derive(Clone)]
pub enum TokenTree<TokenStream, Span, Symbol> {
    Group(Group<TokenStream, Span>),
    Punct(Punct<Span>),
    Ident(Ident<Span, Symbol>),
    Literal(Literal<Span, Symbol>),
}

compound_traits!(
    enum TokenTree<TokenStream, Span, Symbol> {
        Group(tt),
        Punct(tt),
        Ident(tt),
        Literal(tt),
    }
);

#[derive(Clone, Debug)]
pub struct Diagnostic<Span> {
    pub level: Level,
    pub message: String,
    pub spans: Vec<Span>,
    pub children: Vec<Diagnostic<Span>>,
}

compound_traits!(
    struct Diagnostic<Span> { level, message, spans, children }
);

/// Globals provided alongside the initial inputs for a macro expansion.
/// Provides values such as spans which are used frequently to avoid RPC.
#[derive(Clone)]
pub struct ExpnGlobals<Span> {
    pub def_site: Span,
    pub call_site: Span,
    pub mixed_site: Span,
}

compound_traits!(
    struct ExpnGlobals<Span> { def_site, call_site, mixed_site }
);

compound_traits!(
    struct Range<T> { start, end }
);
