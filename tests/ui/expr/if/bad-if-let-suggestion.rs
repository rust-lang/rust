fn a() {
    if let x = 1 && i = 2 {}
    //~^ ERROR cannot find value `i` in this scope
    //~| ERROR mismatched types
    //~| ERROR expected expression, found `let` statement
}

fn b() {
    if (i + j) = i {}
    //~^ ERROR cannot find value `i` in this scope
    //~| ERROR cannot find value `i` in this scope
    //~| ERROR cannot find value `j` in this scope
}

fn c() {
    if x[0] = 1 {}
    //~^ ERROR cannot find value `x` in this scope
}

fn main() {}
