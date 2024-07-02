macro_rules! inner {
    (#![$inner:meta]) => ()
}

inner! {
    /// Outer
} //~^ ERROR no rules expected `[`

fn main() { }
