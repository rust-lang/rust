macro_rules! macro_panic {
    ($not_a_function:expr, $some_argument:ident) => {
        $not_a_function($some_argument)
    }
}

fn main() {
    let mut value_a = 0;
    let mut value_b = 0;
    macro_panic!(value_a, value_b);
    //~^ ERROR expected function, found `{integer}`
}
