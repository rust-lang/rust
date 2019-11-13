fn main() {
    let (x, y) = ();
//~^ ERROR mismatched types
//~| expected type `()`
//~| found tuple `(_, _)`
//~| expected (), found tuple
    return x;
}
