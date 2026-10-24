// Test that the predicate printed in an unresolved method error prints the
// generics for a generic associated type.

trait X {
    type Y<T>;
}

trait M { //~ NOTE
    fn f(&self) {}
}

impl<T: X<Y<i32> = i32>> M for T {}
//~^ NOTE trait bound `<S as X>::Y<i32> = i32` was not satisfied
//~| NOTE
//~| NOTE
//~| NOTE

struct S;
//~^ NOTE method `f` not found for this struct because it doesn't satisfy `<S as X>::Y<i32> = i32` or `S: M`

impl X for S {
    type Y<T> = bool;
}

fn f(a: S) {
    a.f();
    //~^ ERROR the method `f` exists for struct `S`, but its trait bounds were not satisfied
    //~| NOTE method cannot be called on `S` due to unsatisfied trait bounds
}

fn main() {}
