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

use crate::{Delimiter, Level, Spacing};

/// Higher-order macro describing the server RPC API, allowing automatic
/// generation of type-safe Rust APIs, both client-side and server-side.
///
/// `with_api!(MySelf, my_self, my_macro)` expands to:
/// ```rust,ignore (pseudo-code)
/// my_macro! {
///     // ...
///     Literal {
///         // ...
///         fn character(ch: char) -> MySelf::Literal;
///         // ...
///         fn span(my_self: &MySelf::Literal) -> MySelf::Span;
///         fn set_span(my_self: &mut MySelf::Literal, span: MySelf::Span);
///     },
///     // ...
/// }
/// ```
///
/// The first two arguments serve to customize the arguments names
/// and argument/return types, to enable several different usecases:
///
/// If `my_self` is just `self`, then each `fn` signature can be used
/// as-is for a method. If it's anything else (`self_` in practice),
/// then the signatures don't have a special `self` argument, and
/// can, therefore, have a different one introduced.
///
/// If `MySelf` is just `Self`, then the types are only valid inside
/// a trait or a trait impl, where the trait has associated types
/// for each of the API types. If non-associated types are desired,
/// a module name (`self` in practice) can be used instead of `Self`.
macro_rules! with_api {
    ($S:ident, $self:ident, $m:ident) => {
        $m! {
            FreeFunctions {
                fn drop($self: $S::FreeFunctions);
                fn injected_env_var(var: &str) -> Option<String>;
                fn track_env_var(var: &str, value: Option<&str>);
                fn track_path(path: &str);
                fn literal_from_str(s: &str) -> Result<Literal<$S::Span, $S::Symbol>, ()>;
                fn emit_diagnostic(diagnostic: Diagnostic<$S::Span>);
            },
            TokenStream {
                fn drop($self: $S::TokenStream);
                fn clone($self: &$S::TokenStream) -> $S::TokenStream;
                fn is_empty($self: &$S::TokenStream) -> bool;
                fn expand_expr($self: &$S::TokenStream) -> Result<$S::TokenStream, ()>;
                fn from_str(src: &str) -> $S::TokenStream;
                fn to_string($self: &$S::TokenStream) -> String;
                fn from_token_tree(
                    tree: TokenTree<$S::TokenStream, $S::Span, $S::Symbol>,
                ) -> $S::TokenStream;
                fn concat_trees(
                    base: Option<$S::TokenStream>,
                    trees: Vec<TokenTree<$S::TokenStream, $S::Span, $S::Symbol>>,
                ) -> $S::TokenStream;
                fn concat_streams(
                    base: Option<$S::TokenStream>,
                    streams: Vec<$S::TokenStream>,
                ) -> $S::TokenStream;
                fn into_trees(
                    $self: $S::TokenStream
                ) -> Vec<TokenTree<$S::TokenStream, $S::Span, $S::Symbol>>;
            },
            SourceFile {
                fn drop($self: $S::SourceFile);
                fn clone($self: &$S::SourceFile) -> $S::SourceFile;
                fn eq($self: &$S::SourceFile, other: &$S::SourceFile) -> bool;
                fn path($self: &$S::SourceFile) -> String;
                fn is_real($self: &$S::SourceFile) -> bool;
            },
            Span {
                fn debug($self: $S::Span) -> String;
                fn source_file($self: $S::Span) -> $S::SourceFile;
                fn parent($self: $S::Span) -> Option<$S::Span>;
                fn source($self: $S::Span) -> $S::Span;
                fn byte_range($self: $S::Span) -> Range<usize>;
                fn start($self: $S::Span) -> $S::Span;
                fn end($self: $S::Span) -> $S::Span;
                fn line($self: $S::Span) -> usize;
                fn column($self: $S::Span) -> usize;
                fn join($self: $S::Span, other: $S::Span) -> Option<$S::Span>;
                fn subspan($self: $S::Span, start: Bound<usize>, end: Bound<usize>) -> Option<$S::Span>;
                fn resolved_at($self: $S::Span, at: $S::Span) -> $S::Span;
                fn source_text($self: $S::Span) -> Option<String>;
                fn save_span($self: $S::Span) -> usize;
                fn recover_proc_macro_span(id: usize) -> $S::Span;
            },
            Symbol {
                fn normalize_and_validate_ident(string: &str) -> Result<$S::Symbol, ()>;
            },
        }
    };
}

// Similar to `with_api`, but only lists the types requiring handles, and they
// are divided into the two storage categories.
macro_rules! with_api_handle_types {
    ($m:ident) => {
        $m! {
            'owned:
            FreeFunctions,
            TokenStream,
            SourceFile,

            'interned:
            Span,
            // Symbol is handled manually
        }
    };
}

// FIXME(eddyb) this calls `encode` for each argument, but in reverse,
// to match the ordering in `reverse_decode`.
macro_rules! reverse_encode {
    ($writer:ident;) => {};
    ($writer:ident; $first:ident $(, $rest:ident)*) => {
        reverse_encode!($writer; $($rest),*);
        $first.encode(&mut $writer, &mut ());
    }
}

// FIXME(eddyb) this calls `decode` for each argument, but in reverse,
// to avoid borrow conflicts from borrows started by `&mut` arguments.
macro_rules! reverse_decode {
    ($reader:ident, $s:ident;) => {};
    ($reader:ident, $s:ident; $first:ident: $first_ty:ty $(, $rest:ident: $rest_ty:ty)*) => {
        reverse_decode!($reader, $s; $($rest: $rest_ty),*);
        let $first = <$first_ty>::decode(&mut $reader, $s);
    }
}

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
use rpc::{Decode, DecodeMut, Encode, Reader, Writer};

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

    // Prevent Send and Sync impls. `!Send`/`!Sync` is the usual way of doing
    // this, but that requires unstable features. rust-analyzer uses this code
    // and avoids unstable features.
    _marker: marker::PhantomData<*mut ()>,
}

#[forbid(unsafe_code)]
#[allow(non_camel_case_types)]
mod api_tags {
    use super::rpc::{DecodeMut, Encode, Reader, Writer};

    macro_rules! declare_tags {
        ($($name:ident {
            $(fn $method:ident($($arg:ident: $arg_ty:ty),* $(,)?) $(-> $ret_ty:ty)*;)*
        }),* $(,)?) => {
            $(
                pub(super) enum $name {
                    $($method),*
                }
                rpc_encode_decode!(enum $name { $($method),* });
            )*

            pub(super) enum Method {
                $($name($name)),*
            }
            rpc_encode_decode!(enum Method { $($name(m)),* });
        }
    }
    with_api!(self, self, declare_tags);
}

/// Helper to wrap associated types to allow trait impl dispatch.
/// That is, normally a pair of impls for `T::Foo` and `T::Bar`
/// can overlap, but if the impls are, instead, on types like
/// `Marked<T::Foo, Foo>` and `Marked<T::Bar, Bar>`, they can't.
trait Mark {
    type Unmarked;
    fn mark(unmarked: Self::Unmarked) -> Self;
}

/// Unwrap types wrapped by `Mark::mark` (see `Mark` for details).
trait Unmark {
    type Unmarked;
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
}
impl<T, M> Unmark for Marked<T, M> {
    type Unmarked = T;
    fn unmark(self) -> Self::Unmarked {
        self.value
    }
}
impl<'a, T, M> Unmark for &'a Marked<T, M> {
    type Unmarked = &'a T;
    fn unmark(self) -> Self::Unmarked {
        &self.value
    }
}
impl<'a, T, M> Unmark for &'a mut Marked<T, M> {
    type Unmarked = &'a mut T;
    fn unmark(self) -> Self::Unmarked {
        &mut self.value
    }
}

impl<T: Mark> Mark for Vec<T> {
    type Unmarked = Vec<T::Unmarked>;
    fn mark(unmarked: Self::Unmarked) -> Self {
        // Should be a no-op due to std's in-place collect optimizations.
        unmarked.into_iter().map(T::mark).collect()
    }
}
impl<T: Unmark> Unmark for Vec<T> {
    type Unmarked = Vec<T::Unmarked>;
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
            }
            impl Unmark for $ty {
                type Unmarked = Self;
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
    char,
    &'_ [u8],
    &'_ str,
    String,
    u8,
    usize,
    Delimiter,
    LitKind,
    Level,
    Spacing,
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
rpc_encode_decode!(
    enum Spacing {
        Alone,
        Joint,
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
        }

        impl<$($T: Unmark),+> Unmark for $name <$($T),+> {
            type Unmarked = $name <$($T::Unmarked),+>;
            fn unmark(self) -> Self::Unmarked {
                $name {
                    $($field: Unmark::unmark(self.$field)),*
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
        }

        impl<$($T: Unmark),+> Unmark for $name <$($T),+> {
            type Unmarked = $name <$($T::Unmarked),+>;
            fn unmark(self) -> Self::Unmarked {
                match self {
                    $($name::$variant $(($field))? => {
                        $name::$variant $((Unmark::unmark($field)))?
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
