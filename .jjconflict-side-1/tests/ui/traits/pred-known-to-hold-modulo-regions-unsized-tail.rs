// This is a non-regression test for issues #108721 and its duplicate #123275 (hopefully, because
// the test is still convoluted and the ICE is fiddly).
//
// `pred_known_to_hold_modulo_regions` prevented "unexpected unsized tail" ICEs with warp/hyper but
// was unknowingly removed in #120463.

//@ build-pass: the ICE happened in codegen

use std::future::Future;
trait TryFuture: Future {
    type Ok;
}
impl<F, T> TryFuture for F
where
    F: ?Sized + Future<Output = Option<T>>,
{
    type Ok = T;
}
trait Executor {}
struct Exec {}
trait HttpBody {
    type Data;
}
trait ConnStreamExec<F> {}
impl<F> ConnStreamExec<F> for Exec where H2Stream<F>: Send {}
impl<E, F> ConnStreamExec<F> for E where E: Executor {}
struct H2Stream<F> {
    _fut: F,
}
trait NewSvcExec<S, E, W: Watcher<S, E>> {
    fn execute_new_svc(&mut self, _fut: NewSvcTask<S, E, W>) {
        unimplemented!()
    }
}
impl<S, E, W> NewSvcExec<S, E, W> for Exec where W: Watcher<S, E> {}
trait Watcher<S, E> {
    type Future;
}
struct NoopWatcher;
impl<S, E> Watcher<S, E> for NoopWatcher
where
    S: HttpService,
    E: ConnStreamExec<S::Future>,
{
    type Future = Option<<<S as HttpService>::ResBody as HttpBody>::Data>;
}
trait Service<Request> {
    type Response;
    type Future;
}
trait HttpService {
    type ResBody: HttpBody;
    type Future;
}
struct Body {}
impl HttpBody for Body {
    type Data = String;
}
impl<S> HttpService for S
where
    S: Service<(), Response = ()>,
{
    type ResBody = Body;
    type Future = S::Future;
}
trait MakeServiceRef<Target> {
    type ResBody;
    type Service: HttpService<ResBody = Self::ResBody>;
}
impl<T, Target, S, F> MakeServiceRef<Target> for T
where
    T: for<'a> Service<&'a Target, Response = S, Future = F>,
    S: HttpService,
{
    type Service = S;
    type ResBody = S::ResBody;
}
fn make_service_fn<F, Target, Ret>(_f: F) -> MakeServiceFn<F>
where
    F: FnMut(&Target) -> Ret,
    Ret: Future,
{
    unimplemented!()
}
struct MakeServiceFn<F> {
    _func: F,
}
impl<'t, F, Ret, Target, Svc> Service<&'t Target> for MakeServiceFn<F>
where
    F: FnMut(&Target) -> Ret,
    Ret: Future<Output = Option<Svc>>,
{
    type Response = Svc;
    type Future = Option<()>;
}
struct AddrIncoming {}
struct Server<I, S, E> {
    _incoming: I,
    _make_service: S,
    _protocol: E,
}
impl<I, S, E, B> Server<I, S, E>
where
    S: MakeServiceRef<(), ResBody = B>,
    B: HttpBody,
    E: ConnStreamExec<<S::Service as HttpService>::Future>,
    E: NewSvcExec<S::Service, E, NoopWatcher>,
{
    fn serve(&mut self) {
        let fut = NewSvcTask::new();
        self._protocol.execute_new_svc(fut);
    }
}
fn serve<S>(_make_service: S) -> Server<AddrIncoming, S, Exec> {
    unimplemented!()
}
struct NewSvcTask<S, E, W: Watcher<S, E>> {
    _state: State<S, E, W>,
}
struct State<S, E, W: Watcher<S, E>> {
    _fut: W::Future,
}
impl<S, E, W: Watcher<S, E>> NewSvcTask<S, E, W> {
    fn new() -> Self {
        unimplemented!()
    }
}
trait Filter {
    type Extract;
    type Future;
    fn map<F>(self, _fun: F) -> MapFilter<Self, F>
    where
        Self: Sized,
    {
        unimplemented!()
    }
    fn wrap_with<W>(self, _wrapper: W) -> W::Wrapped
    where
        Self: Sized,
        W: Wrap<Self>,
    {
        unimplemented!()
    }
}
fn service<F>(_filter: F) -> FilteredService<F>
where
    F: Filter,
{
    unimplemented!()
}
struct FilteredService<F> {
    _filter: F,
}
impl<F> Service<()> for FilteredService<F>
where
    F: Filter,
{
    type Response = ();
    type Future = FilteredFuture<F::Future>;
}
struct FilteredFuture<F> {
    _fut: F,
}
struct MapFilter<T, F> {
    _filter: T,
    _func: F,
}
impl<T, F> Filter for MapFilter<T, F>
where
    T: Filter,
    F: Func<T::Extract>,
{
    type Extract = F::Output;
    type Future = MapFilterFuture<T, F>;
}
struct MapFilterFuture<T: Filter, F> {
    _extract: T::Future,
    _func: F,
}
trait Wrap<F> {
    type Wrapped;
}
fn make_filter_fn<F, U>(_func: F) -> FilterFn<F>
where
    F: Fn() -> U,
{
    unimplemented!()
}
struct FilterFn<F> {
    _func: F,
}
impl<F, U> Filter for FilterFn<F>
where
    F: Fn() -> U,
    U: TryFuture,
    U::Ok: Send,
{
    type Extract = U::Ok;
    type Future = Option<U>;
}
fn trace<F>(_func: F) -> Trace<F>
where
    F: Fn(),
{
    unimplemented!()
}
struct Trace<F> {
    _func: F,
}
impl<FN, F> Wrap<F> for Trace<FN> {
    type Wrapped = WithTrace<FN, F>;
}
struct WithTrace<FN, F> {
    _filter: F,
    _trace: FN,
}
impl<FN, F> Filter for WithTrace<FN, F>
where
    F: Filter,
{
    type Extract = ();
    type Future = (F::Future, fn(F::Extract));
}
trait Func<Args> {
    type Output;
}
impl<F, R> Func<()> for F
where
    F: Fn() -> R,
{
    type Output = R;
}
fn main() {
    let make_service = make_service_fn(|_| {
        let tracer = trace(|| unimplemented!());
        let filter = make_filter_fn(|| std::future::ready(Some(())))
            .map(|| "Hello, world")
            .wrap_with(tracer);
        let svc = service(filter);
        std::future::ready(Some(svc))
    });
    let mut server = serve(make_service);
    server.serve();
}
