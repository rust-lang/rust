//! Server-side traits.

use super::*;

// FIXME(eddyb) generate the definition of `HandleStore` in `server.rs`.
use super::client::HandleStore;

pub trait Types {
    type FreeFunctions: 'static;
    type TokenStream: 'static + Clone;
    type TokenStreamBuilder: 'static;
    type TokenStreamIter: 'static + Clone;
    type Group: 'static + Clone;
    type Punct: 'static + Copy + Eq + Hash;
    type Ident: 'static + Copy + Eq + Hash;
    type Literal: 'static + Clone;
    type SourceFile: 'static + Clone;
    type MultiSpan: 'static;
    type Diagnostic: 'static;
    type Span: 'static + Copy + Eq + Hash;
}

/// Declare an associated fn of one of the traits below, adding necessary
/// default bodies.
macro_rules! associated_fn {
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
        $(pub trait $name: Types {
            $(associated_fn!(fn $method(&mut self, $($arg: $arg_ty),*) $(-> $ret_ty)?);)*
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
        // FIXME(eddyb) `pub` only for `ExecutionStrategy` below.
        pub trait DispatcherTrait {
            // HACK(eddyb) these are here to allow `Self::$name` to work below.
            $(type $name;)*
            fn dispatch(&mut self, buf: Buffer) -> Buffer;
        }

        impl<S: Server> DispatcherTrait for Dispatcher<MarkedTypes<S>> {
            $(type $name = <MarkedTypes<S> as Types>::$name;)*
            fn dispatch(&mut self, mut buf: Buffer) -> Buffer {
                let Dispatcher { handle_store, server } = self;

                let mut reader = &buf[..];
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

                            buf.clear();
                            r.encode(&mut buf, handle_store);
                        })*
                    }),*
                }
                buf
            }
        }
    }
}
with_api!(Self, self_, define_dispatcher_impl);

pub trait ExecutionStrategy {
    fn run_bridge_and_client(
        &self,
        dispatcher: &mut impl DispatcherTrait,
        input: Buffer,
        run_client: extern "C" fn(Bridge<'_>) -> Buffer,
        force_show_panics: bool,
    ) -> Buffer;
}

pub struct SameThread;

impl ExecutionStrategy for SameThread {
    fn run_bridge_and_client(
        &self,
        dispatcher: &mut impl DispatcherTrait,
        input: Buffer,
        run_client: extern "C" fn(Bridge<'_>) -> Buffer,
        force_show_panics: bool,
    ) -> Buffer {
        let mut dispatch = |buf| dispatcher.dispatch(buf);

        run_client(Bridge {
            cached_buffer: input,
            dispatch: (&mut dispatch).into(),
            force_show_panics,
            _marker: marker::PhantomData,
        })
    }
}

// NOTE(eddyb) Two implementations are provided, the second one is a bit
// faster but neither is anywhere near as fast as same-thread execution.

pub struct CrossThread1;

impl ExecutionStrategy for CrossThread1 {
    fn run_bridge_and_client(
        &self,
        dispatcher: &mut impl DispatcherTrait,
        input: Buffer,
        run_client: extern "C" fn(Bridge<'_>) -> Buffer,
        force_show_panics: bool,
    ) -> Buffer {
        use std::sync::mpsc::channel;

        let (req_tx, req_rx) = channel();
        let (res_tx, res_rx) = channel();

        let join_handle = thread::spawn(move || {
            let mut dispatch = |buf| {
                req_tx.send(buf).unwrap();
                res_rx.recv().unwrap()
            };

            run_client(Bridge {
                cached_buffer: input,
                dispatch: (&mut dispatch).into(),
                force_show_panics,
                _marker: marker::PhantomData,
            })
        });

        for b in req_rx {
            res_tx.send(dispatcher.dispatch(b)).unwrap();
        }

        join_handle.join().unwrap()
    }
}

pub struct CrossThread2;

impl ExecutionStrategy for CrossThread2 {
    fn run_bridge_and_client(
        &self,
        dispatcher: &mut impl DispatcherTrait,
        input: Buffer,
        run_client: extern "C" fn(Bridge<'_>) -> Buffer,
        force_show_panics: bool,
    ) -> Buffer {
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

            let r = run_client(Bridge {
                cached_buffer: input,
                dispatch: (&mut dispatch).into(),
                force_show_panics,
                _marker: marker::PhantomData,
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
>(
    strategy: &impl ExecutionStrategy,
    handle_counters: &'static client::HandleCounters,
    server: S,
    input: I,
    run_client: extern "C" fn(Bridge<'_>) -> Buffer,
    force_show_panics: bool,
) -> Result<O, PanicMessage> {
    let mut dispatcher =
        Dispatcher { handle_store: HandleStore::new(handle_counters), server: MarkedTypes(server) };

    let mut buf = Buffer::new();
    input.encode(&mut buf, &mut dispatcher.handle_store);

    buf = strategy.run_bridge_and_client(&mut dispatcher, buf, run_client, force_show_panics);

    Result::decode(&mut &buf[..], &mut dispatcher.handle_store)
}

impl client::Client<crate::TokenStream, crate::TokenStream> {
    pub fn run<S: Server>(
        &self,
        strategy: &impl ExecutionStrategy,
        server: S,
        input: S::TokenStream,
        force_show_panics: bool,
    ) -> Result<S::TokenStream, PanicMessage> {
        let client::Client { get_handle_counters, run, _marker } = *self;
        run_server(
            strategy,
            get_handle_counters(),
            server,
            <MarkedTypes<S> as Types>::TokenStream::mark(input),
            run,
            force_show_panics,
        )
        .map(<MarkedTypes<S> as Types>::TokenStream::unmark)
    }
}

impl client::Client<(crate::TokenStream, crate::TokenStream), crate::TokenStream> {
    pub fn run<S: Server>(
        &self,
        strategy: &impl ExecutionStrategy,
        server: S,
        input: S::TokenStream,
        input2: S::TokenStream,
        force_show_panics: bool,
    ) -> Result<S::TokenStream, PanicMessage> {
        let client::Client { get_handle_counters, run, _marker } = *self;
        run_server(
            strategy,
            get_handle_counters(),
            server,
            (
                <MarkedTypes<S> as Types>::TokenStream::mark(input),
                <MarkedTypes<S> as Types>::TokenStream::mark(input2),
            ),
            run,
            force_show_panics,
        )
        .map(<MarkedTypes<S> as Types>::TokenStream::unmark)
    }
}
