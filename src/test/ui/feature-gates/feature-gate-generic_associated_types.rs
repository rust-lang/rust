use std::ops::Deref;

trait PointerFamily<U> {
    type Pointer<T>: Deref<Target = T>;
    //~^ ERROR generic associated types are unstable
    type Pointer2<T>: Deref<Target = T> where T: Clone, U: Clone;
    //~^ ERROR generic associated types are unstable
    //~| ERROR where clauses on associated types are unstable
}

struct Foo;

impl PointerFamily<u32> for Foo {
    type Pointer<Usize> = Box<Usize>;
    //~^ ERROR generic associated types are unstable
    type Pointer2<U32> = Box<U32>;
    //~^ ERROR generic associated types are unstable
}

trait Bar {
    type Assoc where Self: Sized;
    //~^ ERROR where clauses on associated types are unstable
}

impl Bar for Foo {
    type Assoc where Self: Sized = Foo;
    //~^ ERROR where clauses on associated types are unstable
}

fn main() {}
