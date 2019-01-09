fn main() {
    let (x, y) = ();
//~^ ERROR mismatched types
//~| expected type `()`
//~| found type `(_, _)`
//~| expected (), found tuple
    return x;
}
