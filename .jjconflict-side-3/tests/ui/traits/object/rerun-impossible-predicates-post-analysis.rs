//@ run-pass

// Regression test for #161441. This is a next-solver bug fixed by #158993
// which affected stable due to `impossible_predicates` already using the next-solver
// by default.

use std::marker::PhantomData;

struct MyError;

trait StreamingBody {
    type BodyError;
}
struct Body;
impl StreamingBody for Body {
    type BodyError = MyError;
}

trait Service {
    type Output;
}
struct HttpClientService;
impl Service for HttpClientService {
    type Output = Body;
}

trait Trait {
    fn method(&self);
}
impl<F, R, ResBody> Trait for (F, PhantomData<R>)
where
    F: Fn() -> R,
    HttpClientService: Service<Output = ResBody>,
    ResBody: StreamingBody<BodyError: Sized>,
{
    fn method(&self) {}
}

fn inspect_websocket_message() -> impl Sized {}

fn main() {
    (&(inspect_websocket_message, PhantomData) as &dyn Trait).method();
}
