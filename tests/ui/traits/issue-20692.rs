trait Array: Sized + Copy {}

fn f<T: Array>(x: &T) {
    let _ = x
    //~^ ERROR `Array` cannot be made into an object
    as
    &dyn Array;
    //~^ ERROR `Array` cannot be made into an object
}

fn main() {}
