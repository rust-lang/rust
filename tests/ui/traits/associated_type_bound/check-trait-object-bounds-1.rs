// Check that we validate associated type bounds for trait objects

trait X {
    type Y: Clone;
}

fn f<T: X + ?Sized>() {
    None::<T::Y>.clone();
}

fn main() {
    f::<dyn X<Y = str>>();
    //~^ ERROR trait `Clone` is not implemented for `str`
}
