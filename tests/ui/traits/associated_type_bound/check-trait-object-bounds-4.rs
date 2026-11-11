// Check that we validate associated type bounds on super traits for trait
// objects

trait Super {
    type Y: Clone;
}

trait X: Super {}

fn f<T: X + ?Sized>() {
    None::<T::Y>.clone();
}

fn main() {
    f::<dyn X<Y = str>>();
    //~^ ERROR the trait bound `str: Clone` is not satisfied
}
