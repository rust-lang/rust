// edition:2018
// run-pass

#![feature(async_await, await_macro, futures_api)]

trait Foo { }

impl Foo for () { }

impl<'a, T> Foo for &'a mut T where T: Foo { }

async fn foo_async<T>(_v: T) -> u8 where T: Foo {
    0
}

async fn bad<T>(v: T) -> u8 where T: Foo {
    await!(foo_async(v))
}

async fn async_main() {
    let mut v = ();

    let _ = await!(bad(&mut v));
    let _ = await!(foo_async(&mut v));
    let _ = await!(bad(v));
}

fn main() {
    let _ = async_main();
}
