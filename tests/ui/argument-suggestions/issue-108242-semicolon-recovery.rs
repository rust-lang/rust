fn foo() {}
fn main() {
    foo(; //~ ERROR this function takes 0 arguments but 2 arguments were supplied
    foo(; //~ ERROR this function takes 0 arguments but 1 argument was supplied
    //~^ ERROR expected one of
} //~ ERROR mismatched closing delimiter
