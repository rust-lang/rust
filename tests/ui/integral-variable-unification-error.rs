fn main() {
    let mut x //~ NOTE expected due to the type of this binding
        =
        2; //~ NOTE expected due to this value
    x = 5.0;
    //~^ ERROR mismatched types
    //~| NOTE expected integer, found floating-point number
}
