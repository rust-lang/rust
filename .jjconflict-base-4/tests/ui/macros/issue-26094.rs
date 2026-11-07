macro_rules! some_macro {
    ($other: expr) => {{
        $other(None) //~ NOTE unexpected argument of type `Option<_>`
    }};
}

fn some_function() {} //~ NOTE defined here

fn main() {
    some_macro!(some_function);
    //~^ ERROR function takes 0 arguments but 1 argument was supplied
}
