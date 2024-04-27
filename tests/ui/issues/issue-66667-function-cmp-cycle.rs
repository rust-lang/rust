fn first() {
    second == 1 //~ ERROR binary operation
    //~^ ERROR mismatched types
    //~| ERROR mismatched types
}

fn second() {
    first == 1 //~ ERROR binary operation
    //~^ ERROR mismatched types
    //~| ERROR mismatched types
}

fn bar() {
    bar == 1 //~ ERROR binary operation
    //~^ ERROR mismatched types
    //~| ERROR mismatched types
}

fn main() {}
