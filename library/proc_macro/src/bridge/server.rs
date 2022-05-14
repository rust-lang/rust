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
    (type TokenStreamBuilder) =>
        (type TokenStreamBuilder: 'static;);
    (type TokenStreamIter) =>
        (type TokenStreamIter: 'static + Clone;);
    (type Group) =>
        (type Group: 'static + Clone;);
    (type Punct) =>
        (type Punct: 'static + Copy + Eq + Hash;);
    (type Ident) =>
        (type Ident: 'static + Copy + Eq + Hash;);
    (type Literal) =>
        (type Literal: 'static + Clone;);
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

macro_rules! declare_server_traits {
    ($($name:ident {
        $(fn $method:ident($($arg:ident: $arg_ty:ty),* $(,)?) $(-> $ret_ty:ty)?;)*
    }),* $(,)?) => {
        pub trait Types {
            $(associated_item!(type $name);)*
        }

        $(pub trait $name: Types {
            $(associated_item!(fn $method(&mut self, $($arg: $arg_ty),*) $(-> $ret_ty)?);)*
        })*

        pub trait Server: Types $(+ $name)* {}
        impl<S: Types $(+ $name)*> Server for S {}
    }
}
with_api!(Self, self_, declare_server_traits);

pub(super) struct MarkedTypes<S: Types>(S);

macro_rules! define_mark_types_impls {
    ($($name:ident {
        $(fn $method:ident($($arg:ident: $arg_ty:ty),* $(,)?) $(-> $ret_ty:ty)?;)*
    }),* $(,)?) => {
        impl<S: Types> Types for MarkedTypes<S> {
            $(type $name = Marked<S::$name, client::$name>;)*
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
        // HACK(eddyb) only a (private) trait because inherent impls can't have
        // associated types (and `Self::AssocType` is used within `$arg_ty`).
        // While `with_api!` allows customizing `Self`, it would have to be
        // extended to allow `Self::` to become `<MarkedTypes<S> as Types>::`.
        trait DispatcherPrivateHelperTrait {
            $(type $name;)*
            fn dispatch(&mut self, b: Buffer<u8>) -> Buffer<u8>;
        }

        impl<S: Server> DispatcherPrivateHelperTrait for Dispatcher<MarkedTypes<S>> {
            $(type $name = <MarkedTypes<S> as Types>::$name;)*
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
        }
    }
}
with_api!(Self, self_, define_dispatcher_impl);

// FIXME(eddyb) a trait alias of `FnMut(Buffer<u8>) -> Buffer<u8>` would allow
// replacing the non-dynamic mentions of that trait, as well.
pub type DynDispatch<'a> = &'a mut dyn FnMut(Buffer<u8>) -> Buffer<u8>;

pub trait ExecutionStrategy {
    fn cross_thread_dispatch(
        &self,
        server_thread_dispatch: impl FnMut(Buffer<u8>) -> Buffer<u8>,
        with_client_thread_dispatch: impl FnOnce(DynDispatch<'_>) -> Buffer<u8> + Send + 'static,
    ) -> Buffer<u8>;
}

pub struct SameThread;

impl ExecutionStrategy for SameThread {
    fn cross_thread_dispatch(
        &self,
        mut server_thread_dispatch: impl FnMut(Buffer<u8>) -> Buffer<u8>,
        with_client_thread_dispatch: impl FnOnce(DynDispatch<'_>) -> Buffer<u8> + Send + 'static,
    ) -> Buffer<u8> {
        with_client_thread_dispatch(&mut server_thread_dispatch)
    }
}

// NOTE(eddyb) Two implementations are provided, the second one is a bit
// faster but neither is anywhere near as fast as same-thread execution.

pub struct CrossThread1;

impl ExecutionStrategy for CrossThread1 {
    fn cross_thread_dispatch(
        &self,
        mut server_thread_dispatch: impl FnMut(Buffer<u8>) -> Buffer<u8>,
        with_client_thread_dispatch: impl FnOnce(DynDispatch<'_>) -> Buffer<u8> + Send + 'static,
    ) -> Buffer<u8> {
        use std::sync::mpsc::channel;

        let (req_tx, req_rx) = channel();
        let (res_tx, res_rx) = channel();

        let join_handle = thread::spawn(move || {
            with_client_thread_dispatch(&mut |b| {
                req_tx.send(b).unwrap();
                res_rx.recv().unwrap()
            })
        });

        for b in req_rx {
            res_tx.send(server_thread_dispatch(b)).unwrap();
        }

        join_handle.join().unwrap()
    }
}

pub struct CrossThread2;

impl ExecutionStrategy for CrossThread2 {
    fn cross_thread_dispatch(
        &self,
        mut server_thread_dispatch: impl FnMut(Buffer<u8>) -> Buffer<u8>,
        with_client_thread_dispatch: impl FnOnce(DynDispatch<'_>) -> Buffer<u8> + Send + 'static,
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
            let r = with_client_thread_dispatch(&mut |b| {
                *state2.lock().unwrap() = State::Req(b);
                server_thread.unpark();
                loop {
                    thread::park();
                    if let State::Res(b) = &mut *state2.lock().unwrap() {
                        break b.take();
                    }
                }
            });

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
            b = server_thread_dispatch(b.take());
            *state.lock().unwrap() = State::Res(b);
            join_handle.thread().unpark();
        }

        join_handle.join().unwrap()
    }
}

fn run_bridge_and_client<D: Copy + Send + 'static>(
    strategy: &impl ExecutionStrategy,
    server_thread_dispatch: impl FnMut(Buffer<u8>) -> Buffer<u8>,
    input: Buffer<u8>,
    run_client: extern "C" fn(Bridge<'_>, D) -> Buffer<u8>,
    client_data: D,
    force_show_panics: bool,
) -> Buffer<u8> {
    enum OptionDynDispatchL {}

    impl<'a> scoped_cell::ApplyL<'a> for OptionDynDispatchL {
        type Out = Option<DynDispatch<'a>>;
    }

    thread_local! {
        /// Dispatch callback held in server TLS, and using server ABI, but
        /// on the client thread (which can differ from the server thread).
        //
        // FIXME(eddyb) how redundant is this with the (also) thread-local
        // client-side `BridgeState`? Some of that indirection can probably
        // be removed, as long as concerns around further isolation methods
        // (IPC and/or wasm) are considered.
        static CLIENT_THREAD_DISPATCH: scoped_cell::ScopedCell<OptionDynDispatchL> =
            scoped_cell::ScopedCell::new(None);
    }

    strategy.cross_thread_dispatch(server_thread_dispatch, move |client_thread_dispatch| {
        CLIENT_THREAD_DISPATCH.with(|dispatch_slot| {
            dispatch_slot.set(Some(client_thread_dispatch), || {
                let mut dispatch = |b| {
                    CLIENT_THREAD_DISPATCH.with(|dispatch_slot| {
                        dispatch_slot.replace(None, |mut client_thread_dispatch| {
                            client_thread_dispatch.as_mut().unwrap()(b)
                        })
                    })
                };

                run_client(
                    Bridge {
                        cached_buffer: input,
                        dispatch: (&mut dispatch).into(),
                        force_show_panics,
                        _marker: marker::PhantomData,
                    },
                    client_data,
                )
            })
        })
    })
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
    run_client: extern "C" fn(Bridge<'_>, D) -> Buffer<u8>,
    client_data: D,
    force_show_panics: bool,
) -> Result<O, PanicMessage> {
    let mut dispatcher =
        Dispatcher { handle_store: HandleStore::new(handle_counters), server: MarkedTypes(server) };

    let mut b = Buffer::new();
    input.encode(&mut b, &mut dispatcher.handle_store);

    b = run_bridge_and_client(
        strategy,
        |b| dispatcher.dispatch(b),
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
