// Check that we validate associated type bounds for trait objects when they
// have bound lifetimes

trait X<'a> {
    type F: FnOnce(&i32) -> &'a i32;
}

fn f<T: for<'r> X<'r> + ?Sized>() {
    None::<T::F>.map(|f| f(&0));
}

fn main() {
    f::<dyn for<'x> X<'x, F = i32>>();
    //~^ ERROR expected a `FnOnce(&i32)` closure, found `i32`
}
