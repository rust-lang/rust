// Determining that `Sub` is dyn-compatible requires keeping track of the `Output = i32` bound,
// which is not yet implemented.

trait Trait {
    type Output;
}

trait Super<T> {
    type Assoc;
}

trait Sub<X: Trait<Output = i32>, T, U>: Super<T, Assoc = X::Output> + Super<U, Assoc = i32> {}

fn check<X: Trait<Output = i32>, T, U>(_: &dyn Sub<X, T, U>) {}
//~^ ERROR the trait `Sub` is not dyn compatible

fn main() {}
