trait Array: Sized + Copy {}

fn f<T: Array>(x: &T) {
    let _ = x
    as
    &dyn Array;
    //~^ ERROR `Array` is not dyn compatible
}

fn main() {}
