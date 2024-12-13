// Regression test for #133941: Don't suggest adding a semicolon during borrowck
// errors when one already exists.

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
        //~^ NOTE first mutable borrow occurs here
        //~| a temporary with access to the first borrow is created here ...
        foo.g(); //~ ERROR cannot borrow `foo` as mutable more than once at a time
        //~^ second mutable borrow occurs here
    };
    //~^ ... and the first borrow might be used here, when that temporary is dropped and runs the destructor for type `Option<Bar<'_>>`
}
