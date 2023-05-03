// edition:2021

fn main() {
    &[1, 2, 3][1:2];
    //~^ ERROR: expected one of
    //~| HELP: you might have meant to make a slice with range index
    //~| HELP: maybe write a path separator here
}