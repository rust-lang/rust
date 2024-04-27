fn main() {
    let (x, y) = ();
//~^ ERROR mismatched types
//~| expected unit type `()`
//~| found tuple `(_, _)`
//~| expected `()`, found
    return x;
}
