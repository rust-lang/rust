// compile-pass
// edition:2018

#![feature(async_await, await_macro)]

trait MyClosure {
    type Args;
}

impl<R> MyClosure for dyn FnMut() -> R
where R: 'static {
    type Args = ();
}

struct MyStream<C: ?Sized + MyClosure> {
    x: C::Args,
}

async fn get_future<C: ?Sized + MyClosure>(_stream: MyStream<C>) {}

async fn f() {
    let messages: MyStream<FnMut()> = unimplemented!();
    await!(get_future(messages));
}

fn main() {}
