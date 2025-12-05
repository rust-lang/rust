struct X;

// Make sure that we show the impl trait refs in the help message with
// their evaluated constants, rather than `core::::array::{impl#30}::{constant#0}`

fn main() {
    <[X; 35] as Default>::default();
    //~^ ERROR the trait bound `[X; 35]: Default` is not satisfied
}
