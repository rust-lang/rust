fn main() {
    match Some(10) {
    //~^ ERROR match arms have incompatible types
    //~| expected type `bool`
    //~| found type `()`
    //~| expected bool, found ()
        Some(5) => false,
        Some(2) => true,
        None    => (),
        _       => true
    }
}
