trait Tr {
    type Out;
}

impl<X> Tr for fn(X) {
    type Out = X;
}

struct S<T: Tr>(&'static S<(T::Out, T::Out)>, std::marker::PhantomData<T>);
//~^ ERROR the trait bound `(<T as Tr>::Out, <T as Tr>::Out): Tr` is not satisfied

const C: &Result<S<fn(())>, ()> = &Err(());

struct S2<T: Tr>(&'static (), std::marker::PhantomData<<T as Tr>::Out>);
struct A2<T>(&'static S2<T>);
//~^ ERROR the trait bound `T: Tr` is not satisfied

const C2: &Result<A2<()>, ()> = &Err(());

fn main() {}
