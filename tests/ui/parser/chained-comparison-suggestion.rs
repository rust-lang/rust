// Check that we get nice suggestions when attempting a chained comparison.

fn comp1() {
    1 < 2 <= 3; //~ ERROR comparison operators cannot be chained
    //~^ ERROR mismatched types
}

fn comp2() {
    1 < 2 < 3; //~ ERROR comparison operators cannot be chained
}

fn comp3() {
    1 <= 2 < 3; //~ ERROR comparison operators cannot be chained
    //~^ ERROR mismatched types
}

fn comp4() {
    1 <= 2 <= 3; //~ ERROR comparison operators cannot be chained
    //~^ ERROR mismatched types
}

fn comp5() {
    1 > 2 >= 3; //~ ERROR comparison operators cannot be chained
    //~^ ERROR mismatched types
}

fn comp6() {
    1 > 2 > 3; //~ ERROR comparison operators cannot be chained
}

fn comp7() {
    1 >= 2 > 3; //~ ERROR comparison operators cannot be chained
}

fn comp8() {
    1 >= 2 >= 3; //~ ERROR comparison operators cannot be chained
    //~^ ERROR mismatched types
}

fn comp9() {
    1 == 2 < 3; //~ ERROR comparison operators cannot be chained
}

fn comp10() {
    1 > 2 == false; //~ ERROR comparison operators cannot be chained
}

fn comp11() {
    1 == 2 == 3; //~ ERROR comparison operators cannot be chained
    //~^ ERROR mismatched types
}

fn main() {}
