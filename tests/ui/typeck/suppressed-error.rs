fn main() {
    let (x, y) = ();
//~^ ERROR mismatched types
//~| NOTE_NONVIRAL expected unit type `()`
//~| NOTE_NONVIRAL found tuple `(_, _)`
//~| NOTE_NONVIRAL expected `()`, found
    return x;
}
