macro_rules! some_macro {
    ($other: expr) => ({
        $other(None) //~ NOTE argument unexpected
    })
}

fn some_function() {} //~ NOTE defined here

fn main() {
    some_macro!(some_function);
    //~^ ERROR this function takes 0 arguments but 1 argument was supplied
    //~| NOTE in this expansion of some_macro!
}
