// Test that the parser detects `&dyn mut`, offers a help message, and
// recovers.

fn main() {
    let r: &dyn mut Trait;
    //~^ ERROR: `mut` must precede `dyn`
    //~| HELP: place `mut` before `dyn`
    //~| ERROR: cannot find trait `Trait` in this scope [E0405]
}
