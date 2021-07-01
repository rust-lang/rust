//! Internal interface for communicating between a `proc_macro` client
//! (a proc macro crate) and a `proc_macro` server (a compiler front-end).
//!
//! Serialization (with C ABI buffers) and unique integer handles are employed
//! to allow safely interfacing between two copies of `proc_macro` built
//! (from the same source) by different compilers with potentially mismatching
//! Rust ABIs (e.g., stage0/bin/rustc vs stage1/bin/rustc during bootstrap).

#![deny(unsafe_code)]

use crate::{Delimiter, Level, LineColumn, Spacing};
use std::fmt;
use std::hash::Hash;
use std::marker;
use std::mem;
use std::ops::Bound;
use std::panic;
use std::sync::atomic::AtomicUsize;
use std::sync::Once;
use std::thread;

/// Higher-order macro describing the server RPC API, allowing automatic
/// generation of type-safe Rust APIs, both client-side and server-side.
///
/// `with_api!(MySelf, my_self, my_macro)` expands to:
/// ```rust,ignore (pseudo-code)
/// my_macro! {
///     // ...
///     Literal {
///         // ...
///         wait fn character(ch: char) -> MySelf::Literal;
///         // ...
///         wait fn span(my_self: &MySelf::Literal) -> MySelf::Span;
///         nowait fn set_span(my_self: &mut MySelf::Literal, span: MySelf::Span);
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
///
/// If the `nowait` modifier is used, the server implementation may not
/// panic, and the client will continue immediately without waiting for
/// a response from the server when in multithreaded mode. If a return
/// type is present, it must be an owning IPC handle. Other return types
/// are not supported with `nowait`.
macro_rules! with_api {
    ($S:ident, $self:ident, $m:ident) => {
        $m! {
            FreeFunctions {
                nowait fn drop($self: $S::FreeFunctions);
                nowait fn track_env_var(var: &str, value: Option<&str>);
            },
            TokenStream {
                nowait fn drop($self: $S::TokenStream);
                nowait fn clone($self: &$S::TokenStream) -> $S::TokenStream;
                wait fn is_empty($self: &$S::TokenStream) -> bool;
                wait fn from_str(src: &str) -> $S::TokenStream;
                wait fn to_string($self: &$S::TokenStream) -> String;
                nowait fn from_token_tree(
                    tree: TokenTree<$S::Group, $S::Punct, $S::Ident, $S::Literal>,
                ) -> $S::TokenStream;
                nowait fn concat_trees(
                    base: Option<$S::TokenStream>,
                    trees: Vec<TokenTree<$S::Group, $S::Punct, $S::Ident, $S::Literal>>,
                ) -> $S::TokenStream;
                nowait fn concat_streams(
                    base: Option<$S::TokenStream>,
                    trees: Vec<$S::TokenStream>,
                ) -> $S::TokenStream;
                wait fn into_iter(
                    $self: $S::TokenStream
                ) -> Vec<TokenTree<$S::Group, $S::Punct, $S::Ident, $S::Literal>>;
            },
            Group {
                nowait fn drop($self: $S::Group);
                nowait fn clone($self: &$S::Group) -> $S::Group;
                nowait fn new(delimiter: Delimiter, stream: Option<$S::TokenStream>) -> $S::Group;
                wait fn delimiter($self: &$S::Group) -> Delimiter;
                nowait fn stream($self: &$S::Group) -> $S::TokenStream;
                wait fn span($self: &$S::Group) -> $S::Span;
                wait fn span_open($self: &$S::Group) -> $S::Span;
                wait fn span_close($self: &$S::Group) -> $S::Span;
                nowait fn set_span($self: &mut $S::Group, span: $S::Span);
            },
            Punct {
                wait fn new(ch: char, spacing: Spacing) -> $S::Punct;
                wait fn as_char($self: $S::Punct) -> char;
                wait fn spacing($self: $S::Punct) -> Spacing;
                wait fn span($self: $S::Punct) -> $S::Span;
                wait fn with_span($self: $S::Punct, span: $S::Span) -> $S::Punct;
            },
            Ident {
                wait fn new(string: &str, span: $S::Span, is_raw: bool) -> $S::Ident;
                wait fn span($self: $S::Ident) -> $S::Span;
                wait fn with_span($self: $S::Ident, span: $S::Span) -> $S::Ident;
            },
            Literal {
                nowait fn drop($self: $S::Literal);
                nowait fn clone($self: &$S::Literal) -> $S::Literal;
                wait fn from_str(s: &str) -> Result<$S::Literal, ()>;
                wait fn debug_kind($self: &$S::Literal) -> String;
                wait fn symbol($self: &$S::Literal) -> String;
                wait fn suffix($self: &$S::Literal) -> Option<String>;
                nowait fn integer(n: &str) -> $S::Literal;
                nowait fn typed_integer(n: &str, kind: &str) -> $S::Literal;
                nowait fn float(n: &str) -> $S::Literal;
                nowait fn f32(n: &str) -> $S::Literal;
                nowait fn f64(n: &str) -> $S::Literal;
                nowait fn string(string: &str) -> $S::Literal;
                nowait fn character(ch: char) -> $S::Literal;
                nowait fn byte_string(bytes: &[u8]) -> $S::Literal;
                wait fn span($self: &$S::Literal) -> $S::Span;
                nowait fn set_span($self: &mut $S::Literal, span: $S::Span);
                wait fn subspan(
                    $self: &$S::Literal,
                    start: Bound<usize>,
                    end: Bound<usize>,
                ) -> Option<$S::Span>;
            },
            SourceFile {
                nowait fn drop($self: $S::SourceFile);
                nowait fn clone($self: &$S::SourceFile) -> $S::SourceFile;
                wait fn eq($self: &$S::SourceFile, other: &$S::SourceFile) -> bool;
                wait fn path($self: &$S::SourceFile) -> String;
                wait fn is_real($self: &$S::SourceFile) -> bool;
            },
            MultiSpan {
                nowait fn drop($self: $S::MultiSpan);
                nowait fn new() -> $S::MultiSpan;
                wait fn push($self: &mut $S::MultiSpan, span: $S::Span);
            },
            Diagnostic {
                wait fn drop($self: $S::Diagnostic);
                wait fn new(level: Level, msg: &str, span: $S::MultiSpan) -> $S::Diagnostic;
                wait fn sub(
                    $self: &mut $S::Diagnostic,
                    level: Level,
                    msg: &str,
                    span: $S::MultiSpan,
                );
                wait fn emit($self: $S::Diagnostic);
            },
            Span {
                wait fn debug($self: $S::Span) -> String;
                wait fn source_file($self: $S::Span) -> $S::SourceFile;
                wait fn parent($self: $S::Span) -> Option<$S::Span>;
                wait fn source($self: $S::Span) -> $S::Span;
                wait fn start($self: $S::Span) -> LineColumn;
                wait fn end($self: $S::Span) -> LineColumn;
                wait fn join($self: $S::Span, other: $S::Span) -> Option<$S::Span>;
                wait fn resolved_at($self: $S::Span, at: $S::Span) -> $S::Span;
                wait fn source_text($self: $S::Span) -> Option<String>;
                wait fn save_span($self: $S::Span) -> usize;
                wait fn recover_proc_macro_span(id: usize) -> $S::Span;
            },
        }
    };
}

// FIXME(eddyb) this calls `encode` for each argument, but in reverse,
// to avoid borrow conflicts from borrows started by `&mut` arguments.
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
mod buffer;
#[forbid(unsafe_code)]
pub mod client;
#[allow(unsafe_code)]
mod closure;
#[forbid(unsafe_code)]
mod handle;
#[macro_use]
#[forbid(unsafe_code)]
mod rpc;
#[allow(unsafe_code)]
mod scoped_cell;
#[forbid(unsafe_code)]
pub mod server;

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
    input: Buffer<u8>,

    /// Server-side function that the client uses to make requests.
    dispatch: closure::Closure<'a, Buffer<u8>, Buffer<u8>>,

    /// If 'true', always invoke the default panic hook
    force_show_panics: bool,
}

impl<'a> !Sync for BridgeConfig<'a> {}
impl<'a> !Send for BridgeConfig<'a> {}

#[forbid(unsafe_code)]
#[allow(non_camel_case_types)]
mod api_tags {
    use super::rpc::{DecodeMut, Encode, Reader, Writer};

    macro_rules! should_wait_impl {
        (wait) => {
            true
        };
        (nowait) => {
            false
        };
    }

    macro_rules! declare_tags {
        ($($name:ident {
            $($wait:ident fn $method:ident($($arg:ident: $arg_ty:ty),* $(,)?) $(-> $ret_ty:ty)*;)*
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

            impl Method {
                pub(super) fn should_wait(&self) -> bool {
                    match self {
                        $($(
                            Method::$name($name::$method) => should_wait_impl!($wait),
                        )*)*
                    }
                }
            }
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
impl<T, M> Unmark for &'a Marked<T, M> {
    type Unmarked = &'a T;
    fn unmark(self) -> Self::Unmarked {
        &self.value
    }
}
impl<T, M> Unmark for &'a mut Marked<T, M> {
    type Unmarked = &'a mut T;
    fn unmark(self) -> Self::Unmarked {
        &mut self.value
    }
}

impl<T: Mark> Mark for Option<T> {
    type Unmarked = Option<T::Unmarked>;
    fn mark(unmarked: Self::Unmarked) -> Self {
        unmarked.map(T::mark)
    }
}
impl<T: Unmark> Unmark for Option<T> {
    type Unmarked = Option<T::Unmarked>;
    fn unmark(self) -> Self::Unmarked {
        self.map(T::unmark)
    }
}

impl<T: Mark, E: Mark> Mark for Result<T, E> {
    type Unmarked = Result<T::Unmarked, E::Unmarked>;
    fn mark(unmarked: Self::Unmarked) -> Self {
        unmarked.map(T::mark).map_err(E::mark)
    }
}
impl<T: Unmark, E: Unmark> Unmark for Result<T, E> {
    type Unmarked = Result<T::Unmarked, E::Unmarked>;
    fn unmark(self) -> Self::Unmarked {
        self.map(T::unmark).map_err(E::unmark)
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
    &'a [u8],
    &'a str,
    String,
    usize,
    Delimiter,
    Level,
    LineColumn,
    Spacing,
    Bound<usize>,
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
rpc_encode_decode!(struct LineColumn { line, column });
rpc_encode_decode!(
    enum Spacing {
        Alone,
        Joint,
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

#[derive(Clone)]
pub enum TokenTree<G, P, I, L> {
    Group(G),
    Punct(P),
    Ident(I),
    Literal(L),
}

compound_traits!(
    enum TokenTree<G, P, I, L> {
        Group(tt),
        Punct(tt),
        Ident(tt),
        Literal(tt),
    }
);

/// Context provided alongside the initial inputs for a macro expansion.
/// Provides values such as spans which are used frequently to avoid RPC.
#[derive(Clone)]
struct ExpnContext<S> {
    def_site: S,
    call_site: S,
    mixed_site: S,
}

compound_traits!(
    struct ExpnContext<Sp> { def_site, call_site, mixed_site }
);
