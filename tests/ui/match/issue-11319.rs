fn main() {
    match Some(10) {
    //~^ NOTE `match` arms have incompatible types
        Some(5) => false,
        //~^ NOTE this is found to be of type `bool`
        Some(2) => true,
        //~^ NOTE this is found to be of type `bool`
        None    => (),
        //~^ ERROR `match` arms have incompatible types
        //~| NOTE expected `bool`, found `()`
        _       => true
    }
}
