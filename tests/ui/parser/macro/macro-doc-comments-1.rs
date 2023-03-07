macro_rules! outer {
    (#[$outer:meta]) => ()
}

outer! {
    //! Inner
} //~^ ERROR no rules expected the token `!`

fn main() { }
