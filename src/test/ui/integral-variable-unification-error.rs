fn main() {
    let mut x = 2;
    x = 5.0;
    //~^ ERROR mismatched types
    //~| expected type `{integer}`
    //~| found type `{float}`
    //~| expected integer, found floating-point number
}
