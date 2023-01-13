fn a() {
    if {}
    //~^ ERROR missing condition for `if` expression
}

fn b() {
    if true && {}
    //~^ ERROR this `if` expression is missing a block after the condition
}

fn c() {
    let x = {};
    if true x
    //~^ ERROR expected `{`, found `x`
}

fn a2() {
    if {} else {}
    //~^ ERROR missing condition for `if` expression
}

fn b2() {
    if true && {} else {}
    //~^ ERROR this `if` expression is missing a block after the condition
}

fn c2() {
    let x = {};
    if true x else {}
    //~^ ERROR expected `{`, found `x`
}

fn d() {
    if true else {}
    //~^ ERROR this `if` expression is missing a block after the condition
}

fn main() {}
