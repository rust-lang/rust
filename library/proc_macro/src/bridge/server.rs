//! Server-side traits.

use std::cell::Cell;
use std::marker::PhantomData;

use super::*;

pub(super) struct HandleStore<S: Server> {
    token_stream: handle::OwnedStore<Marked<S::TokenStream, client::TokenStream>>,
    span: handle::InternedStore<Marked<S::Span, client::Span>>,
}

impl<S: Server> HandleStore<S> {
    fn new(handle_counters: &'static client::HandleCounters) -> Self {
        HandleStore {
            token_stream: handle::OwnedStore::new(&handle_counters.token_stream),
            span: handle::InternedStore::new(&handle_counters.span),
        }
    }
}

impl<S: Server> Encode<HandleStore<S>> for Marked<S::TokenStream, client::TokenStream> {
    fn encode(self, w: &mut Buffer, s: &mut HandleStore<S>) {
        s.token_stream.alloc(self).encode(w, s);
    }
}

impl<S: Server> Decode<'_, '_, HandleStore<S>> for Marked<S::TokenStream, client::TokenStream> {
    fn decode(r: &mut &[u8], s: &mut HandleStore<S>) -> Self {
        s.token_stream.take(handle::Handle::decode(r, &mut ()))
    }
}

impl<'s, S: Server> Decode<'_, 's, HandleStore<S>>
    for &'s Marked<S::TokenStream, client::TokenStream>
{
    fn decode(r: &mut &[u8], s: &'s mut HandleStore<S>) -> Self {
        &s.token_stream[handle::Handle::decode(r, &mut ())]
    }
}

impl<S: Server> Encode<HandleStore<S>> for Marked<S::Span, client::Span> {
    fn encode(self, w: &mut Buffer, s: &mut HandleStore<S>) {
        s.span.alloc(self).encode(w, s);
    }
}

impl<S: Server> Decode<'_, '_, HandleStore<S>> for Marked<S::Span, client::Span> {
    fn decode(r: &mut &[u8], s: &mut HandleStore<S>) -> Self {
        s.span.copy(handle::Handle::decode(r, &mut ()))
    }
}

struct Dispatcher<S: Server> {
    handle_store: HandleStore<S>,
    server: S,
}

macro_rules! define_server_dispatcher_impl {
    (
        $(fn $method:ident($($arg:ident: $arg_ty:ty),* $(,)?) $(-> $ret_ty:ty)*;)*
    ) => {
        pub trait Server {
            type TokenStream: 'static + Clone;
            type Span: 'static + Copy + Eq + Hash;
            type Symbol: 'static;

            fn globals(&mut self) -> ExpnGlobals<Self::Span>;

            /// Intern a symbol received from RPC
            fn intern_symbol(ident: &str) -> Self::Symbol;

            /// Recover the string value of a symbol, and invoke a callback with it.
            fn with_symbol_string(symbol: &Self::Symbol, f: impl FnOnce(&str));

            $(fn $method(&mut self, $($arg: $arg_ty),*) $(-> $ret_ty)?;)*
        }

        // FIXME(eddyb) `pub` only for `ExecutionStrategy` below.
        pub trait DispatcherTrait {
            // HACK(eddyb) these are here to allow `Self::$name` to work below.
            type TokenStream;
            type Span;
            type Symbol;

            fn dispatch(&mut self, buf: Buffer) -> Buffer;
        }

        impl<S: Server> DispatcherTrait for Dispatcher<S> {
            type TokenStream = Marked<S::TokenStream, client::TokenStream>;
            type Span = Marked<S::Span, client::Span>;
            type Symbol = Marked<S::Symbol, client::Symbol>;

            fn dispatch(&mut self, mut buf: Buffer) -> Buffer {
                let Dispatcher { handle_store, server } = self;

                let mut reader = &buf[..];
                match ApiTags::decode(&mut reader, &mut ()) {
                    $(ApiTags::$method => {
                        let mut call_method = || {
                            $(let $arg = <$arg_ty>::decode(&mut reader, handle_store).unmark();)*
                            let r = server.$method($($arg),*);
                            $(
                                let r: $ret_ty = Mark::mark(r);
                            )*
                            r
                        };
                        // HACK(eddyb) don't use `panic::catch_unwind` in a panic.
                        // If client and server happen to use the same `std`,
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
                }
                buf
            }
        }
    }
}
with_api!(Self, define_server_dispatcher_impl);

pub trait ExecutionStrategy {
    fn run_bridge_and_client(
        &self,
        dispatcher: &mut impl DispatcherTrait,
        input: Buffer,
        run_client: extern "C" fn(BridgeConfig<'_>) -> Buffer,
        force_show_panics: bool,
    ) -> Buffer;
}

thread_local! {
    /// While running a proc-macro with the same-thread executor, this flag will
    /// be set, forcing nested proc-macro invocations (e.g. due to
    /// `TokenStream::expand_expr`) to be run using a cross-thread executor.
    ///
    /// This is required as the thread-local state in the proc_macro client does
    /// not handle being re-entered, and will invalidate all `Symbol`s when
    /// entering a nested macro.
    static ALREADY_RUNNING_SAME_THREAD: Cell<bool> = const { Cell::new(false) };
}

/// Keep `ALREADY_RUNNING_SAME_THREAD` (see also its documentation)
/// set to `true`, preventing same-thread reentrance.
struct RunningSameThreadGuard(());

impl RunningSameThreadGuard {
    fn new() -> Self {
        let already_running = ALREADY_RUNNING_SAME_THREAD.replace(true);
        assert!(
            !already_running,
            "same-thread nesting (\"reentrance\") of proc macro executions is not supported"
        );
        RunningSameThreadGuard(())
    }
}

impl Drop for RunningSameThreadGuard {
    fn drop(&mut self) {
        ALREADY_RUNNING_SAME_THREAD.set(false);
    }
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
    P: MessagePipe<Buffer> + Send + 'static,
{
    fn run_bridge_and_client(
        &self,
        dispatcher: &mut impl DispatcherTrait,
        input: Buffer,
        run_client: extern "C" fn(BridgeConfig<'_>) -> Buffer,
        force_show_panics: bool,
    ) -> Buffer {
        if self.cross_thread || ALREADY_RUNNING_SAME_THREAD.get() {
            <CrossThread<P>>::new().run_bridge_and_client(
                dispatcher,
                input,
                run_client,
                force_show_panics,
            )
        } else {
            SameThread.run_bridge_and_client(dispatcher, input, run_client, force_show_panics)
        }
    }
}

pub struct SameThread;

impl ExecutionStrategy for SameThread {
    fn run_bridge_and_client(
        &self,
        dispatcher: &mut impl DispatcherTrait,
        input: Buffer,
        run_client: extern "C" fn(BridgeConfig<'_>) -> Buffer,
        force_show_panics: bool,
    ) -> Buffer {
        let _guard = RunningSameThreadGuard::new();

        let mut dispatch = |buf| dispatcher.dispatch(buf);

        run_client(BridgeConfig { input, dispatch: (&mut dispatch).into(), force_show_panics })
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
    P: MessagePipe<Buffer> + Send + 'static,
{
    fn run_bridge_and_client(
        &self,
        dispatcher: &mut impl DispatcherTrait,
        input: Buffer,
        run_client: extern "C" fn(BridgeConfig<'_>) -> Buffer,
        force_show_panics: bool,
    ) -> Buffer {
        let (mut server, mut client) = P::new();

        let join_handle = thread::spawn(move || {
            let mut dispatch = |b: Buffer| -> Buffer {
                client.send(b);
                client.recv().expect("server died while client waiting for reply")
            };

            run_client(BridgeConfig { input, dispatch: (&mut dispatch).into(), force_show_panics })
        });

        while let Some(b) = server.recv() {
            server.send(dispatcher.dispatch(b));
        }

        join_handle.join().unwrap()
    }
}

/// A message pipe used for communicating between server and client threads.
pub trait MessagePipe<T>: Sized {
    /// Creates a new pair of endpoints for the message pipe.
    fn new() -> (Self, Self);

    /// Send a message to the other endpoint of this pipe.
    fn send(&mut self, value: T);

    /// Receive a message from the other endpoint of this pipe.
    ///
    /// Returns `None` if the other end of the pipe has been destroyed, and no
    /// message was received.
    fn recv(&mut self) -> Option<T>;
}

fn run_server<
    S: Server,
    I: Encode<HandleStore<S>>,
    O: for<'a, 's> Decode<'a, 's, HandleStore<S>>,
>(
    strategy: &impl ExecutionStrategy,
    handle_counters: &'static client::HandleCounters,
    server: S,
    input: I,
    run_client: extern "C" fn(BridgeConfig<'_>) -> Buffer,
    force_show_panics: bool,
) -> Result<O, PanicMessage> {
    let mut dispatcher = Dispatcher { handle_store: HandleStore::new(handle_counters), server };

    let globals = dispatcher.server.globals();

    let mut buf = Buffer::new();
    (<ExpnGlobals<Marked<S::Span, client::Span>> as Mark>::mark(globals), input)
        .encode(&mut buf, &mut dispatcher.handle_store);

    buf = strategy.run_bridge_and_client(&mut dispatcher, buf, run_client, force_show_panics);

    Result::decode(&mut &buf[..], &mut dispatcher.handle_store)
}

impl client::Client<crate::TokenStream, crate::TokenStream> {
    pub fn run<S>(
        &self,
        strategy: &impl ExecutionStrategy,
        server: S,
        input: S::TokenStream,
        force_show_panics: bool,
    ) -> Result<S::TokenStream, PanicMessage>
    where
        S: Server,
        S::TokenStream: Default,
    {
        let client::Client { handle_counters, run, _marker } = *self;
        run_server(
            strategy,
            handle_counters,
            server,
            <Marked<S::TokenStream, client::TokenStream>>::mark(input),
            run,
            force_show_panics,
        )
        .map(|s| {
            <Option<Marked<S::TokenStream, client::TokenStream>>>::unmark(s).unwrap_or_default()
        })
    }
}

impl client::Client<(crate::TokenStream, crate::TokenStream), crate::TokenStream> {
    pub fn run<S>(
        &self,
        strategy: &impl ExecutionStrategy,
        server: S,
        input: S::TokenStream,
        input2: S::TokenStream,
        force_show_panics: bool,
    ) -> Result<S::TokenStream, PanicMessage>
    where
        S: Server,
        S::TokenStream: Default,
    {
        let client::Client { handle_counters, run, _marker } = *self;
        run_server(
            strategy,
            handle_counters,
            server,
            (
                <Marked<S::TokenStream, client::TokenStream>>::mark(input),
                <Marked<S::TokenStream, client::TokenStream>>::mark(input2),
            ),
            run,
            force_show_panics,
        )
        .map(|s| {
            <Option<Marked<S::TokenStream, client::TokenStream>>>::unmark(s).unwrap_or_default()
        })
    }
}
