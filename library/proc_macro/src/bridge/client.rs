//! Client-side types.

use super::*;

use std::marker::PhantomData;

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
            pub(crate) struct $oty {
                handle: handle::Handle,
                // Prevent Send and Sync impls. `!Send`/`!Sync` is the usual
                // way of doing this, but that requires unstable features.
                // rust-analyzer uses this code and avoids unstable features.
                _marker: PhantomData<*mut ()>,
            }

            // Forward `Drop::drop` to the inherent `drop` method.
            impl Drop for $oty {
                fn drop(&mut self) {
                    $oty {
                        handle: self.handle,
                        _marker: PhantomData,
                    }.drop();
                }
            }

            impl<S> Encode<S> for $oty {
                fn encode(self, w: &mut Writer, s: &mut S) {
                    let handle = self.handle;
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
                    self.handle.encode(w, s);
                }
            }

            impl<'s, S: server::Types> Decode<'_, 's, HandleStore<server::MarkedTypes<S>>>
                for &'s Marked<S::$oty, $oty>
            {
                fn decode(r: &mut Reader<'_>, s: &'s HandleStore<server::MarkedTypes<S>>) -> Self {
                    &s.$oty[handle::Handle::decode(r, &mut ())]
                }
            }

            impl<S> Encode<S> for &mut $oty {
                fn encode(self, w: &mut Writer, s: &mut S) {
                    self.handle.encode(w, s);
                }
            }

            impl<'s, S: server::Types> DecodeMut<'_, 's, HandleStore<server::MarkedTypes<S>>>
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
                    $oty {
                        handle: handle::Handle::decode(r, s),
                        _marker: PhantomData,
                    }
                }
            }
        )*

        $(
            #[derive(Copy, Clone, PartialEq, Eq, Hash)]
            pub(crate) struct $ity {
                handle: handle::Handle,
                // Prevent Send and Sync impls. `!Send`/`!Sync` is the usual
                // way of doing this, but that requires unstable features.
                // rust-analyzer uses this code and avoids unstable features.
                _marker: PhantomData<*mut ()>,
            }

            impl<S> Encode<S> for $ity {
                fn encode(self, w: &mut Writer, s: &mut S) {
                    self.handle.encode(w, s);
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
                    $ity {
                        handle: handle::Handle::decode(r, s),
                        _marker: PhantomData,
                    }
                }
            }
        )*
    }
}
define_handles! {
    'owned:
    FreeFunctions,
    TokenStream,
    SourceFile,

    'interned:
    Span,
}

// FIXME(eddyb) generate these impls by pattern-matching on the
// names of methods - also could use the presence of `fn drop`
// to distinguish between 'owned and 'interned, above.
// Alternatively, special "modes" could be listed of types in with_api
// instead of pattern matching on methods, here and in server decl.

impl Clone for TokenStream {
    fn clone(&self) -> Self {
        self.clone()
    }
}

impl Clone for SourceFile {
    fn clone(&self) -> Self {
        self.clone()
    }
}

impl Span {
    pub(crate) fn def_site() -> Span {
        Bridge::with(|bridge| bridge.globals.def_site)
    }

    pub(crate) fn call_site() -> Span {
        Bridge::with(|bridge| bridge.globals.call_site)
    }

    pub(crate) fn mixed_site() -> Span {
        Bridge::with(|bridge| bridge.globals.mixed_site)
    }
}

impl fmt::Debug for Span {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.debug())
    }
}

pub(crate) use super::symbol::Symbol;

macro_rules! define_client_side {
    ($($name:ident {
        $(fn $method:ident($($arg:ident: $arg_ty:ty),* $(,)?) $(-> $ret_ty:ty)*;)*
    }),* $(,)?) => {
        $(impl $name {
            $(pub(crate) fn $method($($arg: $arg_ty),*) $(-> $ret_ty)* {
                Bridge::with(|bridge| {
                    let mut buf = bridge.cached_buffer.take();

                    buf.clear();
                    api_tags::Method::$name(api_tags::$name::$method).encode(&mut buf, &mut ());
                    reverse_encode!(buf; $($arg),*);

                    buf = bridge.dispatch.call(buf);

                    let r = Result::<_, PanicMessage>::decode(&mut &buf[..], &mut ());

                    bridge.cached_buffer = buf;

                    r.unwrap_or_else(|e| panic::resume_unwind(e.into()))
                })
            })*
        })*
    }
}
with_api!(self, self, define_client_side);

struct Bridge<'a> {
    /// Reusable buffer (only `clear`-ed, never shrunk), primarily
    /// used for making requests.
    cached_buffer: Buffer,

    /// Server-side function that the client uses to make requests.
    dispatch: closure::Closure<'a, Buffer, Buffer>,

    /// Provided globals for this macro expansion.
    globals: ExpnGlobals<Span>,
}

impl<'a> !Send for Bridge<'a> {}
impl<'a> !Sync for Bridge<'a> {}

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

pub(crate) fn is_available() -> bool {
    BridgeState::with(|state| match state {
        BridgeState::Connected(_) | BridgeState::InUse => true,
        BridgeState::NotConnected => false,
    })
}

/// A client-side RPC entry-point, which may be using a different `proc_macro`
/// from the one used by the server, but can be invoked compatibly.
///
/// Note that the (phantom) `I` ("input") and `O` ("output") type parameters
/// decorate the `Client<I, O>` with the RPC "interface" of the entry-point, but
/// do not themselves participate in ABI, at all, only facilitate type-checking.
///
/// E.g. `Client<TokenStream, TokenStream>` is the common proc macro interface,
/// used for `#[proc_macro] fn foo(input: TokenStream) -> TokenStream`,
/// indicating that the RPC input and output will be serialized token streams,
/// and forcing the use of APIs that take/return `S::TokenStream`, server-side.
#[repr(C)]
pub struct Client<I, O> {
    // FIXME(eddyb) use a reference to the `static COUNTERS`, instead of
    // a wrapper `fn` pointer, once `const fn` can reference `static`s.
    pub(super) get_handle_counters: extern "C" fn() -> &'static HandleCounters,

    pub(super) run: extern "C" fn(BridgeConfig<'_>) -> Buffer,

    pub(super) _marker: PhantomData<fn(I) -> O>,
}

impl<I, O> Copy for Client<I, O> {}
impl<I, O> Clone for Client<I, O> {
    fn clone(&self) -> Self {
        *self
    }
}

fn maybe_install_panic_hook(force_show_panics: bool) {
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
}

/// Client-side helper for handling client panics, entering the bridge,
/// deserializing input and serializing output.
// FIXME(eddyb) maybe replace `Bridge::enter` with this?
fn run_client<A: for<'a, 's> DecodeMut<'a, 's, ()>, R: Encode<()>>(
    config: BridgeConfig<'_>,
    f: impl FnOnce(A) -> R,
) -> Buffer {
    let BridgeConfig { input: mut buf, dispatch, force_show_panics, .. } = config;

    panic::catch_unwind(panic::AssertUnwindSafe(|| {
        maybe_install_panic_hook(force_show_panics);

        // Make sure the symbol store is empty before decoding inputs.
        Symbol::invalidate_all();

        let reader = &mut &buf[..];
        let (globals, input) = <(ExpnGlobals<Span>, A)>::decode(reader, &mut ());

        // Put the buffer we used for input back in the `Bridge` for requests.
        let new_state =
            BridgeState::Connected(Bridge { cached_buffer: buf.take(), dispatch, globals });

        BRIDGE_STATE.with(|state| {
            state.set(new_state, || {
                let output = f(input);

                // Take the `cached_buffer` back out, for the output value.
                buf = Bridge::with(|bridge| bridge.cached_buffer.take());

                // HACK(eddyb) Separate encoding a success value (`Ok(output)`)
                // from encoding a panic (`Err(e: PanicMessage)`) to avoid
                // having handles outside the `bridge.enter(|| ...)` scope, and
                // to catch panics that could happen while encoding the success.
                //
                // Note that panics should be impossible beyond this point, but
                // this is defensively trying to avoid any accidental panicking
                // reaching the `extern "C"` (which should `abort` but might not
                // at the moment, so this is also potentially preventing UB).
                buf.clear();
                Ok::<_, ()>(output).encode(&mut buf, &mut ());
            })
        })
    }))
    .map_err(PanicMessage::from)
    .unwrap_or_else(|e| {
        buf.clear();
        Err::<(), _>(e).encode(&mut buf, &mut ());
    });

    // Now that a response has been serialized, invalidate all symbols
    // registered with the interner.
    Symbol::invalidate_all();
    buf
}

impl Client<crate::TokenStream, crate::TokenStream> {
    pub const fn expand1(f: impl Fn(crate::TokenStream) -> crate::TokenStream + Copy) -> Self {
        Client {
            get_handle_counters: HandleCounters::get,
            run: super::selfless_reify::reify_to_extern_c_fn_hrt_bridge(move |bridge| {
                run_client(bridge, |input| f(crate::TokenStream(Some(input))).0)
            }),
            _marker: PhantomData,
        }
    }
}

impl Client<(crate::TokenStream, crate::TokenStream), crate::TokenStream> {
    pub const fn expand2(
        f: impl Fn(crate::TokenStream, crate::TokenStream) -> crate::TokenStream + Copy,
    ) -> Self {
        Client {
            get_handle_counters: HandleCounters::get,
            run: super::selfless_reify::reify_to_extern_c_fn_hrt_bridge(move |bridge| {
                run_client(bridge, |(input, input2)| {
                    f(crate::TokenStream(Some(input)), crate::TokenStream(Some(input2))).0
                })
            }),
            _marker: PhantomData,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub enum ProcMacro {
    CustomDerive {
        trait_name: &'static str,
        attributes: &'static [&'static str],
        client: Client<crate::TokenStream, crate::TokenStream>,
    },

    Attr {
        name: &'static str,
        client: Client<(crate::TokenStream, crate::TokenStream), crate::TokenStream>,
    },

    Bang {
        name: &'static str,
        client: Client<crate::TokenStream, crate::TokenStream>,
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

    pub const fn custom_derive(
        trait_name: &'static str,
        attributes: &'static [&'static str],
        expand: impl Fn(crate::TokenStream) -> crate::TokenStream + Copy,
    ) -> Self {
        ProcMacro::CustomDerive { trait_name, attributes, client: Client::expand1(expand) }
    }

    pub const fn attr(
        name: &'static str,
        expand: impl Fn(crate::TokenStream, crate::TokenStream) -> crate::TokenStream + Copy,
    ) -> Self {
        ProcMacro::Attr { name, client: Client::expand2(expand) }
    }

    pub const fn bang(
        name: &'static str,
        expand: impl Fn(crate::TokenStream) -> crate::TokenStream + Copy,
    ) -> Self {
        ProcMacro::Bang { name, client: Client::expand1(expand) }
    }
}
