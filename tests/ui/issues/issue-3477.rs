fn main() {
    let _p: char = 100;
    //~^ ERROR mismatched types
    //~| NOTE expected `char`, found `u8`
    //~| NOTE expected due to this
}
