#![feature(plugin)]
#![plugin(clippy)]

#[deny(bool_comparison)]
fn main() {
    let x = true;
    if x == true { "yes" } else { "no" };
    //~^ ERROR equality checks against true are unnecessary
    //~| HELP try simplifying it as shown:
    //~| SUGGESTION if x { "yes" } else { "no" };
    if x == false { "yes" } else { "no" };
    //~^ ERROR equality checks against false can be replaced by a negation
    //~| HELP try simplifying it as shown:
    //~| SUGGESTION if !x { "yes" } else { "no" };
    if true == x { "yes" } else { "no" };
    //~^ ERROR equality checks against true are unnecessary
    //~| HELP try simplifying it as shown:
    //~| SUGGESTION if x { "yes" } else { "no" };
    if false == x { "yes" } else { "no" };
    //~^ ERROR equality checks against false can be replaced by a negation
    //~| HELP try simplifying it as shown:
    //~| SUGGESTION if !x { "yes" } else { "no" };
}
