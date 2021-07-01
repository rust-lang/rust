//! Server-side traits.

use super::*;

use std::marker::PhantomData;

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

/// Helper methods defined by `Server` types not invoked over RPC.
pub trait Context: Types {
    fn def_site(&mut self) -> Self::Span;
    fn call_site(&mut self) -> Self::Span;
    fn mixed_site(&mut self) -> Self::Span;
}

macro_rules! declare_server_traits {
    ($($name:ident {
        $($wait:ident fn $method:ident($($arg:ident: $arg_ty:ty),* $(,)?) $(-> $ret_ty:ty)?;)*
    }),* $(,)?) => {
        pub trait Types {
            $(associated_item!(type $name);)*
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
}

macro_rules! define_mark_types_impls {
    ($($name:ident {
        $($wait:ident fn $method:ident($($arg:ident: $arg_ty:ty),* $(,)?) $(-> $ret_ty:ty)?;)*
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

pub(super) trait InitOwnedHandle<S> {
    fn init_handle(self, raw_handle: handle::Handle, s: &mut S);
}

struct Dispatcher<S: Types> {
    handle_store: HandleStore<S>,
    server: S,
}

macro_rules! maybe_handle_nowait_reply {
    (wait, $reader:ident, $r:ident, $handle_store:ident, $ret_ty:ty) => {};
    (nowait, $reader:ident, $r:ident, $handle_store:ident, $ret_ty:ty) => {
        let $r = $r.map(|r| {
            let raw_handle = handle::Handle::decode(&mut $reader, &mut ());
            <$ret_ty as InitOwnedHandle<_>>::init_handle(r, raw_handle, $handle_store);
        });
    };
}

macro_rules! define_dispatcher_impl {
    ($($name:ident {
        $($wait:ident fn $method:ident($($arg:ident: $arg_ty:ty),* $(,)?) $(-> $ret_ty:ty)?;)*
    }),* $(,)?) => {
        // FIXME(eddyb) `pub` only for `ExecutionStrategy` below.
        pub trait DispatcherTrait {
            // HACK(eddyb) these are here to allow `Self::$name` to work below.
            $(type $name;)*
            fn dispatch(&mut self, b: Buffer<u8>) -> Buffer<u8>;
        }

        impl<S: Server> DispatcherTrait for Dispatcher<MarkedTypes<S>> {
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

                            $(maybe_handle_nowait_reply!($wait, reader, r, handle_store, $ret_ty);)?

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

pub trait ExecutionStrategy {
    fn run_bridge_and_client<D: Copy + Send + 'static>(
        &self,
        dispatcher: &mut impl DispatcherTrait,
        input: Buffer<u8>,
        run_client: extern "C" fn(BridgeConfig<'_>, D) -> Buffer<u8>,
        client_data: D,
        force_show_panics: bool,
    ) -> Buffer<u8>;
}

pub struct MaybeCrossThread<P> {
    cross_thread: bool,
    marker: PhantomData<P>,
}

impl<P> MaybeCrossThread<P> {
    pub const fn new(cross_thread: bool) -> Self {
        MaybeCrossThread { cross_thread, marker: PhantomData }
    }
}

impl<P> ExecutionStrategy for MaybeCrossThread<P>
where
    P: MessagePipe<Buffer<u8>> + Send + 'static,
{
    fn run_bridge_and_client<D: Copy + Send + 'static>(
        &self,
        dispatcher: &mut impl DispatcherTrait,
        input: Buffer<u8>,
        run_client: extern "C" fn(BridgeConfig<'_>, D) -> Buffer<u8>,
        client_data: D,
        force_show_panics: bool,
    ) -> Buffer<u8> {
        if self.cross_thread {
            <CrossThread<P>>::new().run_bridge_and_client(
                dispatcher,
                input,
                run_client,
                client_data,
                force_show_panics,
            )
        } else {
            SameThread.run_bridge_and_client(
                dispatcher,
                input,
                run_client,
                client_data,
                force_show_panics,
            )
        }
    }
}

#[derive(Default)]
pub struct SameThread;

impl ExecutionStrategy for SameThread {
    fn run_bridge_and_client<D: Copy + Send + 'static>(
        &self,
        dispatcher: &mut impl DispatcherTrait,
        input: Buffer<u8>,
        run_client: extern "C" fn(BridgeConfig<'_>, D) -> Buffer<u8>,
        client_data: D,
        force_show_panics: bool,
    ) -> Buffer<u8> {
        let mut dispatch = |b| dispatcher.dispatch(b);

        run_client(
            BridgeConfig { input, dispatch: (&mut dispatch).into(), force_show_panics },
            client_data,
        )
    }
}

pub struct CrossThread<P>(PhantomData<P>);

impl<P> CrossThread<P> {
    pub const fn new() -> Self {
        CrossThread(PhantomData)
    }
}

impl<P> ExecutionStrategy for CrossThread<P>
where
    P: MessagePipe<Buffer<u8>> + Send + 'static,
{
    fn run_bridge_and_client<D: Copy + Send + 'static>(
        &self,
        dispatcher: &mut impl DispatcherTrait,
        input: Buffer<u8>,
        run_client: extern "C" fn(BridgeConfig<'_>, D) -> Buffer<u8>,
        client_data: D,
        force_show_panics: bool,
    ) -> Buffer<u8> {
        let (mut server, mut client) = P::new();

        let join_handle = thread::spawn(move || {
            let mut dispatch = |b: Buffer<u8>| -> Buffer<u8> {
                let method_tag = api_tags::Method::decode(&mut &b[..], &mut ());
                client.send(b);

                if method_tag.should_wait() {
                    client.recv().expect("server died while client waiting for reply")
                } else {
                    Buffer::new()
                }
            };

            run_client(
                BridgeConfig { input, dispatch: (&mut dispatch).into(), force_show_panics },
                client_data,
            )
        });

        while let Some(b) = server.recv() {
            let method_tag = api_tags::Method::decode(&mut &b[..], &mut ());
            let b = dispatcher.dispatch(b);

            if method_tag.should_wait() {
                server.send(b);
            } else if let Err(err) = <Result<(), PanicMessage>>::decode(&mut &b[..], &mut ()) {
                panic::resume_unwind(err.into());
            }
        }

        join_handle.join().unwrap()
    }
}

/// A message pipe used for communicating between server and client threads.
pub trait MessagePipe<T>: Sized {
    /// Create a new pair of endpoints for the message pipe.
    fn new() -> (Self, Self);

    /// Send a message to the other endpoint of this pipe.
    fn send(&mut self, value: T);

    /// Receive a message from the other endpoint of this pipe.
    ///
    /// Returns `None` if the other end of the pipe has been destroyed, and no
    /// message was received.
    fn recv(&mut self) -> Option<T>;
}

/// Implementation of `MessagePipe` using `std::sync::mpsc`
pub struct StdMessagePipe<T> {
    tx: std::sync::mpsc::Sender<T>,
    rx: std::sync::mpsc::Receiver<T>,
}

impl<T> MessagePipe<T> for StdMessagePipe<T> {
    fn new() -> (Self, Self) {
        let (tx1, rx1) = std::sync::mpsc::channel();
        let (tx2, rx2) = std::sync::mpsc::channel();
        (StdMessagePipe { tx: tx1, rx: rx2 }, StdMessagePipe { tx: tx2, rx: rx1 })
    }

    fn send(&mut self, v: T) {
        self.tx.send(v).unwrap();
    }

    fn recv(&mut self) -> Option<T> {
        self.rx.recv().ok()
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
