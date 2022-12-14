// Test that the predicate printed in an unresolved method error prints the
// generics for a generic associated type.

trait X {
    type Y<T>;
}

trait M {
    fn f(&self) {}
}

impl<T: X<Y<i32> = i32>> M for T {}
//~^ NOTE trait bound `<S as X>::Y<i32> = i32` was not satisfied
//~| NOTE unsatisfied trait bound introduced here
//~| NOTE
//~| NOTE

struct S;
//~^ NOTE method `f` not found for this
//~| NOTE doesn't satisfy `<S as X>::Y<i32> = i32`
//~| NOTE doesn't satisfy `S: M`

impl X for S {
    type Y<T> = bool;
}

fn f(a: S) {
    a.f();
    //~^ ERROR the method `f` exists for struct `S`, but its trait bounds were not satisfied
    //~| NOTE method cannot be called on `S` due to unsatisfied trait bounds
}

fn main() {}
