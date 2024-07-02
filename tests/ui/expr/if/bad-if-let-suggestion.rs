fn a() {
    if let x = 1 && i = 2 {}
    //~^ ERROR cannot find value `i`
    //~| ERROR mismatched types
    //~| ERROR expected expression, found `let` statement
}

fn b() {
    if (i + j) = i {}
    //~^ ERROR cannot find value `i`
    //~| ERROR cannot find value `i`
    //~| ERROR cannot find value `j`
}

fn c() {
    if x[0] = 1 {}
    //~^ ERROR cannot find value `x`
}

fn main() {}
