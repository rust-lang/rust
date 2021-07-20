//! Server-side traits.

use super::*;

// FIXME(eddyb) generate the definition of `HandleStore` in `server.rs`.
use super::client::HandleStore;

/// Declare an associated item of one of the traits below, optionally
/// adjusting it (i.e., adding bounds to types and default bodies to methods).
macro_rules! associated_item {
    (type FreeFunctions) =>
        (type FreeFunctions: 'static;);
    (type TokenStream) =>
        (type TokenStream: 'static + Clone;);
    (type SourceFile) =>
        (type SourceFile: 'static + Clone;);
    (type MultiSpan) =>
        (type MultiSpan: 'static;);
    (type Diagnostic) =>
        (type Diagnostic: 'static;);
    (type Span) =>
        (type Span: 'static + Copy + Eq + Hash;);
    (fn drop(&mut self, $arg:ident: $arg_ty:ty)) =>
        (fn drop(&mut self, $arg: $arg_ty) { mem::drop($arg) });
    (fn clone(&mut self, $arg:ident: $arg_ty:ty) -> $ret_ty:ty) =>
        (fn clone(&mut self, $arg: $arg_ty) -> $ret_ty { $arg.clone() });
    ($($item:tt)*) => ($($item)*;)
}

/// Helper methods defined by `Server` types not invoked over RPC.
pub trait Context: Types {
    fn def_site(&mut self) -> Self::Span;
    fn call_site(&mut self) -> Self::Span;
    fn mixed_site(&mut self) -> Self::Span;

    /// Check if an identifier is valid, and return `Ok(...)` if it is.
    ///
    /// May be called on any thread.
    ///
    /// Returns `Ok(Some(str))` with a normalized version of the identifier if
    /// normalization is required, and `Ok(None)` if the existing identifier is
    /// already normalized.
    fn validate_ident(ident: &str) -> Result<Option<String>, ()>;

    /// Intern a symbol received from RPC
    fn intern_symbol(ident: &str) -> Self::Symbol;

    /// Recover the string value of a symbol, and invoke a callback with it.
    fn with_symbol_string(symbol: &Self::Symbol, f: impl FnOnce(&str));
}

macro_rules! declare_server_traits {
    ($($name:ident {
        $(fn $method:ident($($arg:ident: $arg_ty:ty),* $(,)?) $(-> $ret_ty:ty)?;)*
    }),* $(,)?) => {
        pub trait Types {
            $(associated_item!(type $name);)*
            type Symbol: 'static + Copy + Eq + Hash;
        }

        $(pub trait $name: Types {
            $(associated_item!(fn $method(&mut self, $($arg: $arg_ty),*) $(-> $ret_ty)?);)*
        })*

        pub trait Server: Types + Context $(+ $name)* {}
        impl<S: Types + Context $(+ $name)*> Server for S {}
    }
}
with_api!(Self, self_, declare_server_traits);

pub(super) struct MarkedTypes<S: Types>(S);

impl<S: Context> Context for MarkedTypes<S> {
    fn def_site(&mut self) -> Self::Span {
        <_>::mark(Context::def_site(&mut self.0))
    }
    fn call_site(&mut self) -> Self::Span {
        <_>::mark(Context::call_site(&mut self.0))
    }
    fn mixed_site(&mut self) -> Self::Span {
        <_>::mark(Context::mixed_site(&mut self.0))
    }
    fn validate_ident(ident: &str) -> Result<Option<String>, ()> {
        S::validate_ident(ident)
    }
    fn intern_symbol(ident: &str) -> Self::Symbol {
        <_>::mark(S::intern_symbol(ident))
    }
    fn with_symbol_string(symbol: &Self::Symbol, f: impl FnOnce(&str)) {
        S::with_symbol_string(symbol.unmark(), f)
    }
}

macro_rules! define_mark_types_impls {
    ($($name:ident {
        $(fn $method:ident($($arg:ident: $arg_ty:ty),* $(,)?) $(-> $ret_ty:ty)?;)*
    }),* $(,)?) => {
        impl<S: Types> Types for MarkedTypes<S> {
            $(type $name = Marked<S::$name, client::$name>;)*
            type Symbol = Marked<S::Symbol, client::Symbol>;
        }

        $(impl<S: $name> $name for MarkedTypes<S> {
            $(fn $method(&mut self, $($arg: $arg_ty),*) $(-> $ret_ty)? {
                <_>::mark($name::$method(&mut self.0, $($arg.unmark()),*))
            })*
        })*
    }
}
with_api!(Self, self_, define_mark_types_impls);

struct Dispatcher<S: Types> {
    handle_store: HandleStore<S>,
    server: S,
}

macro_rules! define_dispatcher_impl {
    ($($name:ident {
        $(fn $method:ident($($arg:ident: $arg_ty:ty),* $(,)?) $(-> $ret_ty:ty)?;)*
    }),* $(,)?) => {
        // FIXME(eddyb) `pub` only for `ExecutionStrategy` below.
        pub trait DispatcherTrait {
            // HACK(eddyb) these are here to allow `Self::$name` to work below.
            $(type $name;)*
            type Symbol;

            fn dispatch(&mut self, b: Buffer<u8>) -> Buffer<u8>;
            fn validate_ident(ident: &str) -> Result<Option<String>, ()>;
        }

        impl<S: Server> DispatcherTrait for Dispatcher<MarkedTypes<S>> {
            $(type $name = <MarkedTypes<S> as Types>::$name;)*
            type Symbol = <MarkedTypes<S> as Types>::Symbol;

            fn dispatch(&mut self, mut b: Buffer<u8>) -> Buffer<u8> {
                let Dispatcher { handle_store, server } = self;

                let mut reader = &b[..];
                match api_tags::Method::decode(&mut reader, &mut ()) {
                    $(api_tags::Method::$name(m) => match m {
                        $(api_tags::$name::$method => {
                            let mut call_method = || {
                                reverse_decode!(reader, handle_store; $($arg: $arg_ty),*);
                                $name::$method(server, $($arg),*)
                            };
                            // HACK(eddyb) don't use `panic::catch_unwind` in a panic.
                            // If client and server happen to use the same `libstd`,
                            // `catch_unwind` asserts that the panic counter was 0,
                            // even when the closure passed to it didn't panic.
                            let r = if thread::panicking() {
                                Ok(call_method())
                            } else {
                                panic::catch_unwind(panic::AssertUnwindSafe(call_method))
                                    .map_err(PanicMessage::from)
                            };

                            b.clear();
                            r.encode(&mut b, handle_store);
                        })*
                    }),*
                }
                b
            }
            fn validate_ident(ident: &str) -> Result<Option<String>, ()> {
                S::validate_ident(ident)
            }
        }
    }
}
with_api!(Self, self_, define_dispatcher_impl);

extern "C" fn validate_ident_impl<D: DispatcherTrait>(
    string: buffer::Slice<'_, u8>,
    normalized: &mut Buffer<u8>,
) -> bool {
    match std::str::from_utf8(&string[..]).map_err(|_| ()).and_then(D::validate_ident) {
        Ok(Some(norm)) => {
            *normalized = norm.into_bytes().into();
            true
        }
        Ok(None) => true,
        Err(_) => false,
    }
}

pub trait ExecutionStrategy {
    fn run_bridge_and_client<D: DispatcherTrait, T: Copy + Send + 'static>(
        &self,
        dispatcher: &mut D,
        input: Buffer<u8>,
        run_client: extern "C" fn(BridgeConfig<'_>, T) -> Buffer<u8>,
        client_data: T,
        force_show_panics: bool,
    ) -> Buffer<u8>;
}

pub struct SameThread;

impl ExecutionStrategy for SameThread {
    fn run_bridge_and_client<D: DispatcherTrait, T: Copy + Send + 'static>(
        &self,
        dispatcher: &mut D,
        input: Buffer<u8>,
        run_client: extern "C" fn(BridgeConfig<'_>, T) -> Buffer<u8>,
        client_data: T,
        force_show_panics: bool,
    ) -> Buffer<u8> {
        let mut dispatch = |b| dispatcher.dispatch(b);

        run_client(
            BridgeConfig {
                input,
                dispatch: (&mut dispatch).into(),
                validate_ident: validate_ident_impl::<D>,
                force_show_panics,
            },
            client_data,
        )
    }
}

// NOTE(eddyb) Two implementations are provided, the second one is a bit
// faster but neither is anywhere near as fast as same-thread execution.

pub struct CrossThread1;

impl ExecutionStrategy for CrossThread1 {
    fn run_bridge_and_client<D: DispatcherTrait, T: Copy + Send + 'static>(
        &self,
        dispatcher: &mut D,
        input: Buffer<u8>,
        run_client: extern "C" fn(BridgeConfig<'_>, T) -> Buffer<u8>,
        client_data: T,
        force_show_panics: bool,
    ) -> Buffer<u8> {
        use std::sync::mpsc::channel;

        let (req_tx, req_rx) = channel();
        let (res_tx, res_rx) = channel();

        let join_handle = thread::spawn(move || {
            let mut dispatch = |b| {
                req_tx.send(b).unwrap();
                res_rx.recv().unwrap()
            };

            run_client(
                BridgeConfig {
                    input,
                    dispatch: (&mut dispatch).into(),
                    validate_ident: validate_ident_impl::<D>,
                    force_show_panics,
                },
                client_data,
            )
        });

        for b in req_rx {
            res_tx.send(dispatcher.dispatch(b)).unwrap();
        }

        join_handle.join().unwrap()
    }
}

pub struct CrossThread2;

impl ExecutionStrategy for CrossThread2 {
    fn run_bridge_and_client<D: DispatcherTrait, T: Copy + Send + 'static>(
        &self,
        dispatcher: &mut D,
        input: Buffer<u8>,
        run_client: extern "C" fn(BridgeConfig<'_>, T) -> Buffer<u8>,
        client_data: T,
        force_show_panics: bool,
    ) -> Buffer<u8> {
        use std::sync::{Arc, Mutex};

        enum State<T> {
            Req(T),
            Res(T),
        }

        let mut state = Arc::new(Mutex::new(State::Res(Buffer::new())));

        let server_thread = thread::current();
        let state2 = state.clone();
        let join_handle = thread::spawn(move || {
            let mut dispatch = |b| {
                *state2.lock().unwrap() = State::Req(b);
                server_thread.unpark();
                loop {
                    thread::park();
                    if let State::Res(b) = &mut *state2.lock().unwrap() {
                        break b.take();
                    }
                }
            };

            let r = run_client(
                BridgeConfig {
                    input,
                    dispatch: (&mut dispatch).into(),
                    validate_ident: validate_ident_impl::<D>,
                    force_show_panics,
                },
                client_data,
            );

            // Wake up the server so it can exit the dispatch loop.
            drop(state2);
            server_thread.unpark();

            r
        });

        // Check whether `state2` was dropped, to know when to stop.
        while Arc::get_mut(&mut state).is_none() {
            thread::park();
            let mut b = match &mut *state.lock().unwrap() {
                State::Req(b) => b.take(),
                _ => continue,
            };
            b = dispatcher.dispatch(b.take());
            *state.lock().unwrap() = State::Res(b);
            join_handle.thread().unpark();
        }

        join_handle.join().unwrap()
    }
}

fn run_server<
    S: Server,
    I: Encode<HandleStore<MarkedTypes<S>>>,
    O: for<'a, 's> DecodeMut<'a, 's, HandleStore<MarkedTypes<S>>>,
    D: Copy + Send + 'static,
>(
    strategy: &impl ExecutionStrategy,
    handle_counters: &'static client::HandleCounters,
    server: S,
    input: I,
    run_client: extern "C" fn(BridgeConfig<'_>, D) -> Buffer<u8>,
    client_data: D,
    force_show_panics: bool,
) -> Result<O, PanicMessage> {
    let mut dispatcher =
        Dispatcher { handle_store: HandleStore::new(handle_counters), server: MarkedTypes(server) };

    let expn_context = ExpnContext {
        def_site: dispatcher.server.def_site(),
        call_site: dispatcher.server.call_site(),
        mixed_site: dispatcher.server.mixed_site(),
    };

    let mut b = Buffer::new();
    (input, expn_context).encode(&mut b, &mut dispatcher.handle_store);

    b = strategy.run_bridge_and_client(
        &mut dispatcher,
        b,
        run_client,
        client_data,
        force_show_panics,
    );

    Result::decode(&mut &b[..], &mut dispatcher.handle_store)
}

impl client::Client<fn(crate::TokenStream) -> crate::TokenStream> {
    pub fn run<S: Server>(
        &self,
        strategy: &impl ExecutionStrategy,
        server: S,
        input: S::TokenStream,
        force_show_panics: bool,
    ) -> Result<S::TokenStream, PanicMessage> {
        let client::Client { get_handle_counters, run, f } = *self;
        run_server(
            strategy,
            get_handle_counters(),
            server,
            <MarkedTypes<S> as Types>::TokenStream::mark(input),
            run,
            f,
            force_show_panics,
        )
        .map(<MarkedTypes<S> as Types>::TokenStream::unmark)
    }
}

impl client::Client<fn(crate::TokenStream, crate::TokenStream) -> crate::TokenStream> {
    pub fn run<S: Server>(
        &self,
        strategy: &impl ExecutionStrategy,
        server: S,
        input: S::TokenStream,
        input2: S::TokenStream,
        force_show_panics: bool,
    ) -> Result<S::TokenStream, PanicMessage> {
        let client::Client { get_handle_counters, run, f } = *self;
        run_server(
            strategy,
            get_handle_counters(),
            server,
            (
                <MarkedTypes<S> as Types>::TokenStream::mark(input),
                <MarkedTypes<S> as Types>::TokenStream::mark(input2),
            ),
            run,
            f,
            force_show_panics,
        )
        .map(<MarkedTypes<S> as Types>::TokenStream::unmark)
    }
}
