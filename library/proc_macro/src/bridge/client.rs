//! Client-side types.

use super::*;

use std::cell::RefCell;
use std::marker::PhantomData;
use std::sync::atomic::AtomicU32;

macro_rules! define_client_handles {
    (
        'owned: $($oty:ident,)*
        'interned: $($ity:ident,)*
    ) => {
        #[repr(C)]
        #[allow(non_snake_case)]
        pub(super) struct HandleCounters {
            $(pub(super) $oty: AtomicU32,)*
            $(pub(super) $ity: AtomicU32,)*
        }

        impl HandleCounters {
            // FIXME(eddyb) use a reference to the `static COUNTERS`, instead of
            // a wrapper `fn` pointer, once `const fn` can reference `static`s.
            extern "C" fn get() -> &'static Self {
                static COUNTERS: HandleCounters = HandleCounters {
                    $($oty: AtomicU32::new(1),)*
                    $($ity: AtomicU32::new(1),)*
                };
                &COUNTERS
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
                    mem::ManuallyDrop::new(self).handle.encode(w, s);
                }
            }

            impl<S> Encode<S> for &$oty {
                fn encode(self, w: &mut Writer, s: &mut S) {
                    self.handle.encode(w, s);
                }
            }

            impl<S> Encode<S> for &mut $oty {
                fn encode(self, w: &mut Writer, s: &mut S) {
                    self.handle.encode(w, s);
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
with_api_handle_types!(define_client_handles);

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
        $(fn $method:ident($($arg:ident: $arg_ty:ty),* $(,)?) $(-> $ret_ty:ty)?;)*
    }),* $(,)?) => {
        $(impl $name {
            $(pub(crate) fn $method($($arg: $arg_ty),*) $(-> $ret_ty)? {
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

#[allow(unsafe_code)]
mod state {
    use super::Bridge;
    use std::cell::{Cell, RefCell};
    use std::ptr;

    thread_local! {
        static BRIDGE_STATE: Cell<*const ()> = const { Cell::new(ptr::null()) };
    }

    pub(super) fn set<'bridge, R>(state: &RefCell<Bridge<'bridge>>, f: impl FnOnce() -> R) -> R {
        struct RestoreOnDrop(*const ());
        impl Drop for RestoreOnDrop {
            fn drop(&mut self) {
                BRIDGE_STATE.set(self.0);
            }
        }

        let inner = ptr::from_ref(state).cast();
        let outer = BRIDGE_STATE.replace(inner);
        let _restore = RestoreOnDrop(outer);

        f()
    }

    pub(super) fn with<R>(
        f: impl for<'bridge> FnOnce(Option<&RefCell<Bridge<'bridge>>>) -> R,
    ) -> R {
        let state = BRIDGE_STATE.get();
        // SAFETY: the only place where the pointer is set is in `set`. It puts
        // back the previous value after the inner call has returned, so we know
        // that as long as the pointer is not null, it came from a reference to
        // a `RefCell<Bridge>` that outlasts the call to this function. Since `f`
        // works the same for any lifetime of the bridge, including the actual
        // one, we can lie here and say that the lifetime is `'static` without
        // anyone noticing.
        let bridge = unsafe { state.cast::<RefCell<Bridge<'static>>>().as_ref() };
        f(bridge)
    }
}

impl Bridge<'_> {
    fn with<R>(f: impl FnOnce(&mut Bridge<'_>) -> R) -> R {
        state::with(|state| {
            let bridge = state.expect("procedural macro API is used outside of a procedural macro");
            let mut bridge = bridge
                .try_borrow_mut()
                .expect("procedural macro API is used while it's already in use");
            f(&mut bridge)
        })
    }
}

pub(crate) fn is_available() -> bool {
    state::with(|s| s.is_some())
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
    // NB. the server can't do this because it may use a different std.
    static HIDE_PANICS_DURING_EXPANSION: Once = Once::new();
    HIDE_PANICS_DURING_EXPANSION.call_once(|| {
        let prev = panic::take_hook();
        panic::set_hook(Box::new(move |info| {
            // We normally report panics by catching unwinds and passing the payload from the
            // unwind back to the compiler, but if the panic doesn't unwind we'll abort before
            // the compiler has a chance to print an error. So we special-case PanicInfo where
            // can_unwind is false.
            if force_show_panics || !is_available() || !info.can_unwind() {
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
        let state = RefCell::new(Bridge { cached_buffer: buf.take(), dispatch, globals });

        let output = state::set(&state, || f(input));

        // Take the `cached_buffer` back out, for the output value.
        buf = RefCell::into_inner(state).cached_buffer;

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
