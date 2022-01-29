#![deny(unused_attributes)] //~ NOTE lint level is defined here

#[cold]
//~^ ERROR attribute should be applied to a function
//~| WARN this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
struct Foo; //~ NOTE not a function

fn main() {
    #[cold]
    //~^ ERROR attribute should be applied to a function
    //~| WARN this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
    5; //~ NOTE not a function
}
