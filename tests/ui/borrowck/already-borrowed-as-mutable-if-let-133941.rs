// https://github.com/rust-lang/rust/issues/133941
use std::marker::PhantomData;

struct Bar<'a>(PhantomData<&'a mut i32>);

impl<'a> Drop for Bar<'a> {
    fn drop(&mut self) {}
}

struct Foo();

impl Foo {
    fn f(&mut self) -> Option<Bar<'_>> {
        None
    }

    fn g(&mut self) {}
}

fn main() {
    let mut foo = Foo();
    while let Some(_) = foo.f() {
        //~^ HELP matches!
        foo.g();
        //~^ ERROR [E0499]
    }
    if let Some(_) = foo.f() {
        //~^ HELP matches!
        foo.g();
        //~^ ERROR [E0499]
    }
    while let Some(_x) = foo.f() {
        foo.g();
        //~^ ERROR [E0499]
    }
    if let Some(_x) = foo.f() {
        foo.g();
        //~^ ERROR [E0499]
    }
    while let Some(_x) = {let _x = foo.f(); foo.g(); None::<()>} {
        //~^ ERROR [E0499]
    }
    if let Some(_x) = {let _x = foo.f(); foo.g(); None::<()>} {
        //~^ ERROR [E0499]
    }
}
