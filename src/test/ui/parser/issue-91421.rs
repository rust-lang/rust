// Regression test for issue #91421.

fn main() {
    let value = if true && {
    //~^ ERROR: this `if` expression has a condition, but no block
    //~| HELP: maybe you forgot the right operand of the condition?
        3
        //~^ ERROR: mismatched types [E0308]
    } else { 4 };
}
