macro_rules! some_macro {
    ($other: expr) => ({
        $other(None)
        //~^ this function takes 0 parameters but 1 parameter was supplied
    })
}

fn some_function() {}

fn main() {
    some_macro!(some_function);
    //~^ in this expansion of some_macro!
}
