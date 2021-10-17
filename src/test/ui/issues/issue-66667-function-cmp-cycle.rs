fn first() {
    second == 1 //~ ERROR can't compare
    //~^ ERROR mismatched types
}

fn second() {
    first == 1 //~ ERROR can't compare
    //~^ ERROR mismatched types
}

fn bar() {
    bar == 1 //~ ERROR can't compare
    //~^ ERROR mismatched types
}

fn main() {}
