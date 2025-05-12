macro_rules! outer {
    (#[$outer:meta]) => ()
}

outer! {
    //! Inner
} //~^ ERROR no rules expected `!`

fn main() { }
