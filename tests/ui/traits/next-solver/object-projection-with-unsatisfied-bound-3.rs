//@compile-flags: -Znext-solver

// Regression test for #155761.
// Previously we assume that projection with object self_ty always has the corresponding
// existential projection predicate which can be used to normalize the projection itself.
// But when the projection on the trait object isn't well-formed. The matching projection
// can be filtered out.
trait Foo {
    type V;
}

trait Callback<T: Foo + ?Sized>: Fn(&T::V) {}

fn try_object<T: Foo>() {
    <dyn Callback<dyn Callback<T>>>::call(());
//~^ ERROR: the trait bound `dyn Callback<T>: Foo` is not satisfied [E0277]
//~| ERROR: the trait bound `dyn Callback<T>: Foo` is not satisfied [E0277]
//~| ERROR: use of unstable library feature `fn_traits` [E0658]
//~| ERROR: the trait bound `dyn Callback<T>: Foo` is not satisfied [E0277]
//~| ERROR: expected a `FnMut<_>` closure, found `dyn Callback<dyn Callback<T>, Output = ()>` [E0277]
//~| ERROR: expected a `FnOnce(&_)` closure, found `dyn Callback<dyn Callback<T>, Output = ()>` [E0277]
//~| ERROR: this function takes 2 arguments but 1 argument was supplied [E0061]
}

fn main() {}
