// Regression test for #61320
// This is the same issue as #61311, just a larger test case.

// build-pass (FIXME(62277): could be check-pass?)

pub struct AndThen<A, B, F>
where
    A: Future,
    B: IntoFuture,
{
    state: (A, B::Future, F),
}

pub struct FutureResult<T, E> {
    inner: Option<Result<T, E>>,
}

impl<T, E> Future for FutureResult<T, E> {
    type Item = T;
    type Error = E;

    fn poll(&mut self) -> Poll<T, E> {
        unimplemented!()
    }
}

pub type Poll<T, E> = Result<T, E>;

impl<A, B, F> Future for AndThen<A, B, F>
where
    A: Future,
    B: IntoFuture<Error = A::Error>,
    F: FnOnce(A::Item) -> B,
{
    type Item = B::Item;
    type Error = B::Error;

    fn poll(&mut self) -> Poll<B::Item, B::Error> {
        unimplemented!()
    }
}

pub trait Future {
    type Item;

    type Error;

    fn poll(&mut self) -> Poll<Self::Item, Self::Error>;

    fn and_then<F, B>(self, f: F) -> AndThen<Self, B, F>
    where
        F: FnOnce(Self::Item) -> B,
        B: IntoFuture<Error = Self::Error>,
        Self: Sized,
    {
        unimplemented!()
    }
}

pub trait IntoFuture {
    /// The future that this type can be converted into.
    type Future: Future<Item = Self::Item, Error = Self::Error>;

    /// The item that the future may resolve with.
    type Item;
    /// The error that the future may resolve with.
    type Error;

    /// Consumes this object and produces a future.
    fn into_future(self) -> Self::Future;
}

impl<F: Future> IntoFuture for F {
    type Future = F;
    type Item = F::Item;
    type Error = F::Error;

    fn into_future(self) -> F {
        self
    }
}

impl<F: ?Sized + Future> Future for ::std::boxed::Box<F> {
    type Item = F::Item;
    type Error = F::Error;

    fn poll(&mut self) -> Poll<Self::Item, Self::Error> {
        (**self).poll()
    }
}

impl<T, E> IntoFuture for Result<T, E> {
    type Future = FutureResult<T, E>;
    type Item = T;
    type Error = E;

    fn into_future(self) -> FutureResult<T, E> {
        unimplemented!()
    }
}

struct Request<T>(T);

trait RequestContext {}
impl<T> RequestContext for T {}
struct NoContext;
impl AsRef<NoContext> for NoContext {
    fn as_ref(&self) -> &Self {
        &NoContext
    }
}

type BoxedError = Box<dyn std::error::Error + Send + Sync>;
type DefaultFuture<T, E> = Box<dyn Future<Item = T, Error = E> + Send>;

trait Guard: Sized {
    type Result: IntoFuture<Item = Self, Error = BoxedError>;
    fn from_request(request: &Request<()>) -> Self::Result;
}

trait FromRequest: Sized {
    type Context;
    type Future: Future<Item = Self, Error = BoxedError> + Send;
    fn from_request(request: Request<()>) -> Self::Future;
}

struct MyGuard;
impl Guard for MyGuard {
    type Result = Result<Self, BoxedError>;
    fn from_request(_request: &Request<()>) -> Self::Result {
        Ok(MyGuard)
    }
}

struct Generic<I> {
    _inner: I,
}

impl<I> FromRequest for Generic<I>
where
    MyGuard: Guard,
    <MyGuard as Guard>::Result: IntoFuture<Item = MyGuard, Error = BoxedError>,
    <<MyGuard as Guard>::Result as IntoFuture>::Future: Send,
    I: FromRequest<Context = NoContext>,
{
    type Future = DefaultFuture<Self, BoxedError>;
    type Context = NoContext;
    fn from_request(headers: Request<()>) -> DefaultFuture<Self, BoxedError> {
        let _future = <MyGuard as Guard>::from_request(&headers)
            .into_future()
            .and_then(move |_| {
                <I as FromRequest>::from_request(headers)
                    .into_future()
                    .and_then(move |fld_inner| Ok(Generic { _inner: fld_inner }).into_future())
            });
        panic!();
    }
}

fn main() {}
