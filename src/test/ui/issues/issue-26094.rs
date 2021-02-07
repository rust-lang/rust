macro_rules! some_macro {
    ($other: expr) => ({
        $other(None) //~ NOTE supplied 1 argument
    })
}

fn some_function() {} //~ NOTE defined here

fn main() {
    some_macro!(some_function);
    //~^ ERROR arguments to this function are incorrect
    //~| NOTE argument unexpected
}
