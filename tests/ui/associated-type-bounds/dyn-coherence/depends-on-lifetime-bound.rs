// Determining that `Sub` is dyn-compatible requires keeping track of the `'a: 'static` bound,
// which is not yet implemented.

trait Super<T> {
    type Assoc;
}

trait Sub<'a: 'static, T, U>: Super<T, Assoc = &'a ()> + Super<U, Assoc = &'static ()> {}

fn foo<'a: 'static, T, U>(_: &dyn Sub<'a, T, U>) {}
//~^ ERROR the trait `Sub` is not dyn compatible

fn main() {}
