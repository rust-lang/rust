//@ known-bug: #148511
//@ edition: 2021
use std::any::Any;

fn main() {
    use_service(make_service());
    let _future = async {};
}

fn make_service() -> impl FooService<Box<dyn Any>, Response: Body> {}

fn use_service<S, R>(_service: S)
where
    S: FooService<R>,
    <S::Response as Body>::Data: Any,
{
}

trait Service<Request> {
    type Response: Body;
}

impl<T, Request> Service<Request> for T {
    type Response = ();
}

trait FooService<Request>: Service<Request> {}

impl<T, Request, Resp> FooService<Request> for T
where
    T: Service<Request, Response = Resp>,
    Resp: Body,
{
}

trait Body {
    type Data;
}

impl<T> Body for T {
    type Data = ();
}
