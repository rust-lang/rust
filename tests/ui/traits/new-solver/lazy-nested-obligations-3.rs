// check-pass
// compile-flags: -Ztrait-solver=next
// Issue 96750

use std::marker::PhantomData;

trait AsyncFn<Arg> {
    type Output;
}
trait RequestFamily {
    type Type<'a>;
}
trait Service {}

struct MyFn;
impl AsyncFn<String> for MyFn {
    type Output = ();
}

impl RequestFamily for String {
    type Type<'a> = String;
}

struct ServiceFromAsyncFn<F, Req>(F, PhantomData<Req>);

impl<F, Req, O> Service for ServiceFromAsyncFn<F, Req>
where
    Req: RequestFamily,
    F: AsyncFn<Req>,
    F: for<'a> AsyncFn<Req::Type<'a>, Output = O>,
{
}

fn assert_service() -> impl Service {
    ServiceFromAsyncFn(MyFn, PhantomData)
}

fn main() {}
