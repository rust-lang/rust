// Make sure that the dyn-compatibility check for coherent associated types
// work correctly when the associated type bounds come from the supertrait.

// Case #1:
// Both the conflicting bounds come from a supertrait.

trait OneSuper<T> {
    type Assoc;
}
trait OneSub<T, U>: OneSuper<T, Assoc = u32> + OneSuper<U, Assoc = u64> {}
trait OneSubSub<T, U>: OneSub<T, U> {}

fn one_check<T, U>(_: &dyn OneSubSub<T, U>) {}
//~^ ERROR the trait `OneSubSub` is not dyn compatible

// Case #2:
// One of the conflicting bounds come from a supertrait, and one is direct.

trait TwoSuper<T> {
    type Assoc;
}
trait TwoSub<T>: TwoSuper<T, Assoc = u32> {}
trait TwoSubSub<T, U>: TwoSub<T> + TwoSuper<U, Assoc = u64> {}

fn two_check<T, U>(_: &dyn TwoSubSub<T, U>) {}
//~^ ERROR the trait `TwoSubSub` is not dyn compatible

fn main() {}
