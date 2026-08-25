//@ compile-flags: -Z trace-macros

#![recursion_limit = "5"]

fn main() {
    macro_rules! stack {
        ($overflow:expr) => {
            print!(stack!($overflow))
            //~^ ERROR recursion limit reached while expanding
            //~| ERROR format argument must be a string literal
        };
    }

    stack!("overflow");
}
