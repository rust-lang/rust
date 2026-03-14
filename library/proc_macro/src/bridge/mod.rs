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
use std::rc::Rc;
use std::sync::Once;
use std::{fmt, marker, panic, thread};

use crate::{Delimiter, Level};

/// Higher-order macro describing the server RPC API, allowing automatic
/// generation of type-safe Rust APIs, both client-side and server-side.
///
/// `with_api!(my_macro, MySpan, MySymbol)` expands to:
/// ```rust,ignore (pseudo-code)
/// my_macro! {
///     fn normalize(string: &str) -> MySymbol;
///     fn span_debug(span: &MySpan) -> String;
///     // ...
/// }
/// ```
///
/// The second (`Span`) and third (`Symbol`)
/// argument serve to customize the argument/return types that need
/// special handling, to enable several different representations of
/// these types.
macro_rules! with_api {
    ($m:ident, $Span: path, $Symbol: path) => {
        $m! {
            fn injected_env_var(var: &str) -> Option<String>;
            fn track_env_var(var: &str, value: Option<&str>);
            fn track_path(path: &str);
            fn literal_from_str(s: &str) -> Result<Literal<$Span, $Symbol>, String>;
            fn emit_diagnostic(diagnostic: Diagnostic<$Span>);

            fn ts_expand_expr(stream: TokenStream<$Span, $Symbol>) -> Result<TokenStream<$Span, $Symbol>, ()>;
            fn ts_from_str(src: &str) -> Result<TokenStream<$Span, $Symbol>, String>;
            fn ts_to_string(stream: TokenStream<$Span, $Symbol>) -> String;

            fn span_debug(span: $Span) -> String;
            fn span_parent(span: $Span) -> Option<$Span>;
            fn span_source(span: $Span) -> $Span;
            fn span_byte_range(span: $Span) -> Range<usize>;
            fn span_start(span: $Span) -> $Span;
            fn span_end(span: $Span) -> $Span;
            fn span_line(span: $Span) -> usize;
            fn span_column(span: $Span) -> usize;
            fn span_file(span: $Span) -> String;
            fn span_local_file(span: $Span) -> Option<String>;
            fn span_join(span: $Span, other: $Span) -> Option<$Span>;
            fn span_subspan(span: $Span, start: Bound<usize>, end: Bound<usize>) -> Option<$Span>;
            fn span_resolved_at(span: $Span, at: $Span) -> $Span;
            fn span_source_text(span: $Span) -> Option<String>;
            fn span_save_span(span: $Span) -> usize;
            fn span_recover_proc_macro_span(id: usize) -> $Span;

            fn symbol_normalize_and_validate_ident(string: &str) -> Result<$Symbol, ()>;
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
    dispatch: closure::Closure<'a>,

    /// If 'true', always invoke the default panic hook
    force_show_panics: bool,
}

impl !Send for BridgeConfig<'_> {}
impl !Sync for BridgeConfig<'_> {}

macro_rules! declare_tags {
    (
        $(fn $method:ident($($arg:ident: $arg_ty:ty),* $(,)?) $(-> $ret_ty:ty)?;)*
    ) => {
        #[allow(non_camel_case_types)]
        pub(super) enum ApiTags {
            $($method),*
        }
        rpc_encode_decode!(enum ApiTags { $($method),* });
    }
}
with_api!(declare_tags, __, __);

/// Helper to wrap associated types to allow trait impl dispatch.
/// That is, normally a pair of impls for `T::Foo` and `T::Bar`
/// can overlap, but if the impls are, instead, on types like
/// `Marked<T::Foo, Foo>` and `Marked<T::Bar, Bar>`, they can't.
trait Mark: Clone {
    type Unmarked: Clone;
    fn mark(unmarked: Self::Unmarked) -> Self;
    fn unmark(self) -> Self::Unmarked;
}

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
struct Marked<T, M> {
    value: T,
    _marker: marker::PhantomData<M>,
}

impl<T: Clone, M: Clone> Mark for Marked<T, M> {
    type Unmarked = T;
    fn mark(unmarked: Self::Unmarked) -> Self {
        Marked { value: unmarked, _marker: marker::PhantomData }
    }
    fn unmark(self) -> Self::Unmarked {
        self.value
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

impl<T: Mark> Mark for Rc<T> {
    type Unmarked = Rc<T::Unmarked>;
    fn mark(unmarked: Self::Unmarked) -> Self {
        Rc::new(Mark::mark(Rc::unwrap_or_clone(unmarked)))
    }
    fn unmark(self) -> Self::Unmarked {
        Rc::new(Mark::unmark(Rc::unwrap_or_clone(self)))
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
    Bound<usize>,
    Range<usize>,
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

rpc_encode_decode!(
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
pub struct Group<Span, Symbol> {
    pub delimiter: Delimiter,
    pub stream: Option<TokenStream<Span, Symbol>>,
    pub span: DelimSpan<Span>,
}

compound_traits!(struct Group<Span, Symbol> { delimiter, stream, span });

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

compound_traits!(struct Literal<Span, Symbol> { kind, symbol, suffix, span });

#[derive(Clone)]
pub enum TokenTree<Span, Symbol> {
    Group(Group<Span, Symbol>),
    Punct(Punct<Span>),
    Ident(Ident<Span, Symbol>),
    Literal(Literal<Span, Symbol>),
}

compound_traits!(
    enum TokenTree<Span, Symbol> {
        Group(tt),
        Punct(tt),
        Ident(tt),
        Literal(tt),
    }
);

#[derive(Clone)]
pub struct TokenStream<Span, Symbol> {
    pub trees: Rc<Vec<TokenTree<Span, Symbol>>>,
}

impl<Span, Symbol> Default for TokenStream<Span, Symbol> {
    fn default() -> Self {
        Self { trees: Rc::new(Vec::new()) }
    }
}

impl<Span, Symbol> TokenStream<Span, Symbol> {
    pub fn new(tts: Vec<TokenTree<Span, Symbol>>) -> Self {
        Self { trees: Rc::new(tts) }
    }
}

compound_traits!(
    struct TokenStream<Span, Symbol> { trees }
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

rpc_encode_decode!(
    struct Range<T> { start, end }
);
