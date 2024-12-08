//@ edition:2018
//@ run-pass

trait Foo { }

impl Foo for () { }

impl<'a, T> Foo for &'a mut T where T: Foo { }

async fn foo_async<T>(_v: T) -> u8 where T: Foo {
    0
}

async fn bad<T>(v: T) -> u8 where T: Foo {
    foo_async(v).await
}

async fn async_main() {
    let mut v = ();

    let _ = bad(&mut v).await;
    let _ = foo_async(&mut v).await;
    let _ = bad(v).await;
}

fn main() {
    let _ = async_main();
}
