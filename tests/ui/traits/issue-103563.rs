// build-pass

fn main() {
    let mut log_service = LogService { inner: Inner };
    log_service.call(());
}

pub trait Service<Request> {
    type Response;

    fn call(&mut self, req: Request) -> Self::Response;
}

pub struct LogService<S> {
    inner: S,
}

impl<T, U, S> Service<T> for LogService<S>
where
    S: Service<T, Response = U>,
    U: Extension + 'static,
    for<'a> U::Item<'a>: std::fmt::Debug,
{
    type Response = S::Response;

    fn call(&mut self, req: T) -> Self::Response {
        self.inner.call(req)
    }
}

pub struct Inner;

impl Service<()> for Inner {
    type Response = Resp;

    fn call(&mut self, req: ()) -> Self::Response {
        Resp::A(req)
    }
}

pub trait Extension {
    type Item<'a>;

    fn touch<F>(self, f: F) -> Self
    where
        for<'a> F: Fn(Self::Item<'a>);
}

pub enum Resp {
    A(()),
}

impl Extension for Resp {
    type Item<'a> = RespItem<'a>;
    fn touch<F>(self, _f: F) -> Self
    where
        for<'a> F: Fn(Self::Item<'a>),
    {
        match self {
            Self::A(a) => Self::A(a),
        }
    }
}

pub enum RespItem<'a> {
    A(&'a ()),
}

impl<'a> std::fmt::Debug for RespItem<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::A(arg0) => f.debug_tuple("A").field(arg0).finish(),
        }
    }
}
