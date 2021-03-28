//! Client-side types.

use super::*;

macro_rules! define_handles {
    (
        'owned: $($oty:ident,)*
        'interned: $($ity:ident,)*
    ) => {
        #[repr(C)]
        #[allow(non_snake_case)]
        pub struct HandleCounters {
            $($oty: AtomicUsize,)*
            $($ity: AtomicUsize,)*
        }

        impl HandleCounters {
            // FIXME(eddyb) use a reference to the `static COUNTERS`, instead of
            // a wrapper `fn` pointer, once `const fn` can reference `static`s.
            extern "C" fn get() -> &'static Self {
                static COUNTERS: HandleCounters = HandleCounters {
                    $($oty: AtomicUsize::new(1),)*
                    $($ity: AtomicUsize::new(1),)*
                };
                &COUNTERS
            }
        }

        // FIXME(eddyb) generate the definition of `HandleStore` in `server.rs`.
        #[repr(C)]
        #[allow(non_snake_case)]
        pub(super) struct HandleStore<S: server::Types> {
            $($oty: handle::OwnedStore<S::$oty>,)*
            $($ity: handle::InternedStore<S::$ity>,)*
        }

        impl<S: server::Types> HandleStore<S> {
            pub(super) fn new(handle_counters: &'static HandleCounters) -> Self {
                HandleStore {
                    $($oty: handle::OwnedStore::new(&handle_counters.$oty),)*
                    $($ity: handle::InternedStore::new(&handle_counters.$ity),)*
                }
            }
        }

        $(
            #[repr(C)]
            pub(crate) struct $oty(handle::Handle);
            impl !Send for $oty {}
            impl !Sync for $oty {}

            // Forward `Drop::drop` to the inherent `drop` method.
            impl Drop for $oty {
                fn drop(&mut self) {
                    $oty(self.0).drop();
                }
            }

            impl<S> Encode<S> for $oty {
                fn encode(self, w: &mut Writer, s: &mut S) {
                    let handle = self.0;
                    mem::forget(self);
                    handle.encode(w, s);
                }
            }

            impl<S: server::Types> DecodeMut<'_, '_, HandleStore<server::MarkedTypes<S>>>
                for Marked<S::$oty, $oty>
            {
                fn decode(r: &mut Reader<'_>, s: &mut HandleStore<server::MarkedTypes<S>>) -> Self {
                    s.$oty.take(handle::Handle::decode(r, &mut ()))
                }
            }

            impl<S> Encode<S> for &$oty {
                fn encode(self, w: &mut Writer, s: &mut S) {
                    self.0.encode(w, s);
                }
            }

            impl<S: server::Types> Decode<'_, 's, HandleStore<server::MarkedTypes<S>>>
                for &'s Marked<S::$oty, $oty>
            {
                fn decode(r: &mut Reader<'_>, s: &'s HandleStore<server::MarkedTypes<S>>) -> Self {
                    &s.$oty[handle::Handle::decode(r, &mut ())]
                }
            }

            impl<S> Encode<S> for &mut $oty {
                fn encode(self, w: &mut Writer, s: &mut S) {
                    self.0.encode(w, s);
                }
            }

            impl<S: server::Types> DecodeMut<'_, 's, HandleStore<server::MarkedTypes<S>>>
                for &'s mut Marked<S::$oty, $oty>
            {
                fn decode(
                    r: &mut Reader<'_>,
                    s: &'s mut HandleStore<server::MarkedTypes<S>>
                ) -> Self {
                    &mut s.$oty[handle::Handle::decode(r, &mut ())]
                }
            }

            impl<S: server::Types> Encode<HandleStore<server::MarkedTypes<S>>>
                for Marked<S::$oty, $oty>
            {
                fn encode(self, w: &mut Writer, s: &mut HandleStore<server::MarkedTypes<S>>) {
                    s.$oty.alloc(self).encode(w, s);
                }
            }

            impl<S> DecodeMut<'_, '_, S> for $oty {
                fn decode(r: &mut Reader<'_>, s: &mut S) -> Self {
                    $oty(handle::Handle::decode(r, s))
                }
            }
        )*

        $(
            #[repr(C)]
            #[derive(Copy, Clone, PartialEq, Eq, Hash)]
            pub(crate) struct $ity(handle::Handle);
            impl !Send for $ity {}
            impl !Sync for $ity {}

            impl<S> Encode<S> for $ity {
                fn encode(self, w: &mut Writer, s: &mut S) {
                    self.0.encode(w, s);
                }
            }

            impl<S: server::Types> DecodeMut<'_, '_, HandleStore<server::MarkedTypes<S>>>
                for Marked<S::$ity, $ity>
            {
                fn decode(r: &mut Reader<'_>, s: &mut HandleStore<server::MarkedTypes<S>>) -> Self {
                    s.$ity.copy(handle::Handle::decode(r, &mut ()))
                }
            }

            impl<S: server::Types> Encode<HandleStore<server::MarkedTypes<S>>>
                for Marked<S::$ity, $ity>
            {
                fn encode(self, w: &mut Writer, s: &mut HandleStore<server::MarkedTypes<S>>) {
                    s.$ity.alloc(self).encode(w, s);
                }
            }

            impl<S> DecodeMut<'_, '_, S> for $ity {
                fn decode(r: &mut Reader<'_>, s: &mut S) -> Self {
                    $ity(handle::Handle::decode(r, s))
                }
            }
        )*
    }
}
define_handles! {
    'owned:
    FreeFunctions,
    TokenStream,
    TokenStreamBuilder,
    TokenStreamIter,
    Group,
    Literal,
    SourceFile,
    MultiSpan,
    Diagnostic,

    'interned:
    Punct,
    Ident,
    Span,
}

// FIXME(eddyb) generate these impls by pattern-matching on the
// names of methods - also could use the presence of `fn drop`
// to distinguish between 'owned and 'interned, above.
// Alternatively, special 'modes" could be listed of types in with_api
// instead of pattern matching on methods, here and in server decl.

impl Clone for TokenStream {
    fn clone(&self) -> Self {
        self.clone()
    }
}

impl Clone for TokenStreamIter {
    fn clone(&self) -> Self {
        self.clone()
    }
}

impl Clone for Group {
    fn clone(&self) -> Self {
        self.clone()
    }
}

impl Clone for Literal {
    fn clone(&self) -> Self {
        self.clone()
    }
}

impl fmt::Debug for Literal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Literal")
            // format the kind without quotes, as in `kind: Float`
            .field("kind", &format_args!("{}", &self.debug_kind()))
            .field("symbol", &self.symbol())
            // format `Some("...")` on one line even in {:#?} mode
            .field("suffix", &format_args!("{:?}", &self.suffix()))
            .field("span", &self.span())
            .finish()
    }
}

impl Clone for SourceFile {
    fn clone(&self) -> Self {
        self.clone()
    }
}

impl fmt::Debug for Span {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.debug())
    }
}

macro_rules! define_client_side {
    ($($name:ident {
        $(fn $method:ident($($arg:ident: $arg_ty:ty),* $(,)?) $(-> $ret_ty:ty)*;)*
    }),* $(,)?) => {
        $(impl $name {
            $(pub(crate) fn $method($($arg: $arg_ty),*) $(-> $ret_ty)* {
                Bridge::with(|bridge| {
                    let mut b = bridge.cached_buffer.take();

                    b.clear();
                    api_tags::Method::$name(api_tags::$name::$method).encode(&mut b, &mut ());
                    reverse_encode!(b; $($arg),*);

                    b = bridge.dispatch.call(b);

                    let r = Result::<_, PanicMessage>::decode(&mut &b[..], &mut ());

                    bridge.cached_buffer = b;

                    r.unwrap_or_else(|e| panic::resume_unwind(e.into()))
                })
            })*
        })*
    }
}
with_api!(self, self, define_client_side);

enum BridgeState<'a> {
    /// No server is currently connected to this client.
    NotConnected,

    /// A server is connected and available for requests.
    Connected(Bridge<'a>),

    /// Access to the bridge is being exclusively acquired
    /// (e.g., during `BridgeState::with`).
    InUse,
}

enum BridgeStateL {}

impl<'a> scoped_cell::ApplyL<'a> for BridgeStateL {
    type Out = BridgeState<'a>;
}

thread_local! {
    static BRIDGE_STATE: scoped_cell::ScopedCell<BridgeStateL> =
        scoped_cell::ScopedCell::new(BridgeState::NotConnected);
}

impl BridgeState<'_> {
    /// Take exclusive control of the thread-local
    /// `BridgeState`, and pass it to `f`, mutably.
    /// The state will be restored after `f` exits, even
    /// by panic, including modifications made to it by `f`.
    ///
    /// N.B., while `f` is running, the thread-local state
    /// is `BridgeState::InUse`.
    fn with<R>(f: impl FnOnce(&mut BridgeState<'_>) -> R) -> R {
        BRIDGE_STATE.with(|state| {
            state.replace(BridgeState::InUse, |mut state| {
                // FIXME(#52812) pass `f` directly to `replace` when `RefMutL` is gone
                f(&mut *state)
            })
        })
    }
}

impl Bridge<'_> {
    pub(crate) fn is_available() -> bool {
        BridgeState::with(|state| match state {
            BridgeState::Connected(_) | BridgeState::InUse => true,
            BridgeState::NotConnected => false,
        })
    }

    fn enter<R>(self, f: impl FnOnce() -> R) -> R {
        let force_show_panics = self.force_show_panics;
        // Hide the default panic output within `proc_macro` expansions.
        // NB. the server can't do this because it may use a different libstd.
        static HIDE_PANICS_DURING_EXPANSION: Once = Once::new();
        HIDE_PANICS_DURING_EXPANSION.call_once(|| {
            let prev = panic::take_hook();
            panic::set_hook(Box::new(move |info| {
                let show = BridgeState::with(|state| match state {
                    BridgeState::NotConnected => true,
                    BridgeState::Connected(_) | BridgeState::InUse => force_show_panics,
                });
                if show {
                    prev(info)
                }
            }));
        });

        BRIDGE_STATE.with(|state| state.set(BridgeState::Connected(self), f))
    }

    fn with<R>(f: impl FnOnce(&mut Bridge<'_>) -> R) -> R {
        BridgeState::with(|state| match state {
            BridgeState::NotConnected => {
                panic!("procedural macro API is used outside of a procedural macro");
            }
            BridgeState::InUse => {
                panic!("procedural macro API is used while it's already in use");
            }
            BridgeState::Connected(bridge) => f(bridge),
        })
    }
}

/// A client-side "global object" (usually a function pointer),
/// which may be using a different `proc_macro` from the one
/// used by the server, but can be interacted with compatibly.
///
/// N.B., `F` must have FFI-friendly memory layout (e.g., a pointer).
/// The call ABI of function pointers used for `F` doesn't
/// need to match between server and client, since it's only
/// passed between them and (eventually) called by the client.
#[repr(C)]
#[derive(Copy, Clone)]
pub struct Client<F> {
    // FIXME(eddyb) use a reference to the `static COUNTERS`, instead of
    // a wrapper `fn` pointer, once `const fn` can reference `static`s.
    pub(super) get_handle_counters: extern "C" fn() -> &'static HandleCounters,
    pub(super) run: extern "C" fn(Bridge<'_>, F) -> Buffer<u8>,
    pub(super) f: F,
}

/// Client-side helper for handling client panics, entering the bridge,
/// deserializing input and serializing output.
// FIXME(eddyb) maybe replace `Bridge::enter` with this?
fn run_client<A: for<'a, 's> DecodeMut<'a, 's, ()>, R: Encode<()>>(
    mut bridge: Bridge<'_>,
    f: impl FnOnce(A) -> R,
) -> Buffer<u8> {
    // The initial `cached_buffer` contains the input.
    let mut b = bridge.cached_buffer.take();

    panic::catch_unwind(panic::AssertUnwindSafe(|| {
        bridge.enter(|| {
            let reader = &mut &b[..];
            let input = A::decode(reader, &mut ());

            // Put the `cached_buffer` back in the `Bridge`, for requests.
            Bridge::with(|bridge| bridge.cached_buffer = b.take());

            let output = f(input);

            // Take the `cached_buffer` back out, for the output value.
            b = Bridge::with(|bridge| bridge.cached_buffer.take());

            // HACK(eddyb) Separate encoding a success value (`Ok(output)`)
            // from encoding a panic (`Err(e: PanicMessage)`) to avoid
            // having handles outside the `bridge.enter(|| ...)` scope, and
            // to catch panics that could happen while encoding the success.
            //
            // Note that panics should be impossible beyond this point, but
            // this is defensively trying to avoid any accidental panicking
            // reaching the `extern "C"` (which should `abort` but might not
            // at the moment, so this is also potentially preventing UB).
            b.clear();
            Ok::<_, ()>(output).encode(&mut b, &mut ());
        })
    }))
    .map_err(PanicMessage::from)
    .unwrap_or_else(|e| {
        b.clear();
        Err::<(), _>(e).encode(&mut b, &mut ());
    });
    b
}

impl Client<fn(crate::TokenStream) -> crate::TokenStream> {
    #[rustc_allow_const_fn_unstable(const_fn)]
    pub const fn expand1(f: fn(crate::TokenStream) -> crate::TokenStream) -> Self {
        extern "C" fn run(
            bridge: Bridge<'_>,
            f: impl FnOnce(crate::TokenStream) -> crate::TokenStream,
        ) -> Buffer<u8> {
            run_client(bridge, |input| f(crate::TokenStream(input)).0)
        }
        Client { get_handle_counters: HandleCounters::get, run, f }
    }
}

impl Client<fn(crate::TokenStream, crate::TokenStream) -> crate::TokenStream> {
    #[rustc_allow_const_fn_unstable(const_fn)]
    pub const fn expand2(
        f: fn(crate::TokenStream, crate::TokenStream) -> crate::TokenStream,
    ) -> Self {
        extern "C" fn run(
            bridge: Bridge<'_>,
            f: impl FnOnce(crate::TokenStream, crate::TokenStream) -> crate::TokenStream,
        ) -> Buffer<u8> {
            run_client(bridge, |(input, input2)| {
                f(crate::TokenStream(input), crate::TokenStream(input2)).0
            })
        }
        Client { get_handle_counters: HandleCounters::get, run, f }
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub enum ProcMacro {
    CustomDerive {
        trait_name: &'static str,
        attributes: &'static [&'static str],
        client: Client<fn(crate::TokenStream) -> crate::TokenStream>,
    },

    Attr {
        name: &'static str,
        client: Client<fn(crate::TokenStream, crate::TokenStream) -> crate::TokenStream>,
    },

    Bang {
        name: &'static str,
        client: Client<fn(crate::TokenStream) -> crate::TokenStream>,
    },
}

impl ProcMacro {
    pub fn name(&self) -> &'static str {
        match self {
            ProcMacro::CustomDerive { trait_name, .. } => trait_name,
            ProcMacro::Attr { name, .. } => name,
            ProcMacro::Bang { name, .. } => name,
        }
    }

    #[rustc_allow_const_fn_unstable(const_fn)]
    pub const fn custom_derive(
        trait_name: &'static str,
        attributes: &'static [&'static str],
        expand: fn(crate::TokenStream) -> crate::TokenStream,
    ) -> Self {
        ProcMacro::CustomDerive { trait_name, attributes, client: Client::expand1(expand) }
    }

    #[rustc_allow_const_fn_unstable(const_fn)]
    pub const fn attr(
        name: &'static str,
        expand: fn(crate::TokenStream, crate::TokenStream) -> crate::TokenStream,
    ) -> Self {
        ProcMacro::Attr { name, client: Client::expand2(expand) }
    }

    #[rustc_allow_const_fn_unstable(const_fn)]
    pub const fn bang(
        name: &'static str,
        expand: fn(crate::TokenStream) -> crate::TokenStream,
    ) -> Self {
        ProcMacro::Bang { name, client: Client::expand1(expand) }
    }
}
