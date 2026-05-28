// Regression test for correct pretty-printing of an AST representing `&(mut x)` in help
// suggestion diagnostic.

fn main() {
    let mut &x = &0;
    //~^ ERROR `mut` must be attached to each individual binding
    //~| HELP add `mut` to each binding
    //~| SUGGESTION &(mut x)
}
