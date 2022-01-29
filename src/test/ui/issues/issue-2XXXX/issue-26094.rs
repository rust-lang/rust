macro_rules! some_macro {
    ($other: expr) => ({
        $other(None) //~ NOTE supplied 1 argument
    })
}

fn some_function() {} //~ NOTE defined here

fn main() {
    some_macro!(some_function);
    //~^ ERROR this function takes 0 arguments but 1 argument was supplied
    //~| NOTE expected 0 arguments
}
