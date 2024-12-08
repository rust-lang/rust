//@ edition:2021

fn main() {
    &[1, 2, 3][1:2];
    //~^ ERROR: expected one of
    //~| HELP: you might have meant a range expression
}
