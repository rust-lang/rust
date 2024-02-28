// #66353: ICE when trying to recover from incorrect associated type

trait _Func<T> {
    fn func(_: Self);
}

trait _A {
    type AssocT;
}

fn main() {
    _Func::< <() as _A>::AssocT >::func(());
    //~^ ERROR trait `_A` is not implemented for `()`
    //~| ERROR trait `_Func<_>` is not implemented for `()`
}
