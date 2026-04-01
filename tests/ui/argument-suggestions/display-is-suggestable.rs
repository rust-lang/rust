use std::fmt::Display;

fn foo(x: &(dyn Display + Send)) {}

fn main() {
    foo();
    //~^ ERROR function takes 1 argument but 0 arguments were supplied
}
