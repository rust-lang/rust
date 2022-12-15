use std::fmt::Display;

fn foo(x: &(dyn Display + Send)) {}

fn main() {
    foo();
    //~^ ERROR this function takes 1 argument but 0 arguments were supplied
}
