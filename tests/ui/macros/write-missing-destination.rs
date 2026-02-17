// Check that `write!` without a destination gives a helpful error message.
// See https://github.com/rust-lang/rust/issues/152493

fn main() {
    write!("S");
    //~^ ERROR requires a destination and format arguments
}
