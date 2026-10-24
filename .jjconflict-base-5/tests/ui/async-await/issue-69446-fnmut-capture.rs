// Regression test for issue #69446 - we should display
// which variable is captured
//@ edition:2018

use core::future::Future;

struct Foo;
impl Foo {
    fn foo(&mut self) {}
}

async fn bar<T>(_: impl FnMut() -> T)
where
    T: Future<Output = ()>,
{}

fn main() {
    let mut x = Foo;
    bar(move || async { //~ ERROR captured
        x.foo();
    });
}
