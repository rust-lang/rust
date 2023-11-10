// revisions: current next
//[next] check-pass
//[next] compile-flags: -Ztrait-solver=next

pub trait Foo<TY> {
    type FooAssociatedTy;
}

pub trait Bar {
    type BarAssociatedTy;

    fn send<T>()
    where
        T: Foo<Self::BarAssociatedTy>,
        T::FooAssociatedTy: Send;
}

impl Bar for () {
    type BarAssociatedTy = ();

    fn send<T>()
    //[current]~^ ERROR the trait bound `T: Foo<()>` is not
    //[current]~| ERROR the trait bound `T: Foo<()>` is not
    //[current]~| ERROR the trait bound `T: Foo<()>` is not
    where
        T: Foo<Self::BarAssociatedTy>,
        //[current]~^ ERROR impl has stricter requirements than tra
        T::FooAssociatedTy: Send,
    {
    }
}

impl Bar for i32 {
    type BarAssociatedTy = ();

    fn send<T>()
    //[current]~^ ERROR the trait bound `T: Foo<()>` is not
    where
        T: Foo<()>,
        //[current]~^ ERROR impl has stricter requirements than tra
        T::FooAssociatedTy: Send,
    {
    }
}

fn main() {
}
