#![feature(plugin)]
#![plugin(clippy)]

#[deny(bool_comparison)]
fn main() {
    let x = true;
    if x == true { "yes" } else { "no" };
    //~^ ERROR equality checks against booleans are unnecesary
    //~| HELP try simplifying it:
    //~| SUGGESTION x
    if x == false { "yes" } else { "no" };
    //~^ ERROR equality checks against booleans are unnecesary
    //~| HELP try simplifying it:
    //~| SUGGESTION !x
    if true == x { "yes" } else { "no" };
    //~^ ERROR equality checks against booleans are unnecesary
    //~| HELP try simplifying it:
    //~| SUGGESTION x
    if false == x { "yes" } else { "no" };
    //~^ ERROR equality checks against booleans are unnecesary
    //~| HELP try simplifying it:
    //~| SUGGESTION !x
}
