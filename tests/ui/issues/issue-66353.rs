// #66353: ICE when trying to recover from incorrect associated type

trait _Func<T> {
    fn func(_: Self);
}

trait _A {
    type AssocT;
}

fn main() {
    _Func::< <() as _A>::AssocT >::func(());
    //~^ ERROR the trait bound `(): _A` is not satisfied
    //~| ERROR the trait bound `(): _Func<_>` is not satisfied
}
