fn main() {
    let (x, y) = ();
//~^ ERROR mismatched types
//~| NOTE expected unit type `()`
//~| NOTE found tuple `(_, _)`
//~| NOTE expected `()`, found
//~| NOTE this expression has type `()`
    return x;
}
