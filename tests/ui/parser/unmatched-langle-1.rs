// Check that a suggestion is issued if there are too many `<`s in a
// generic argument list, and that the parser recovers properly.

fn main() {
    foo::<<<<Ty<i32>>();
    //~^ ERROR: unmatched angle brackets
    //~| ERROR: cannot find function `foo` in this scope [E0425]
    //~| ERROR: cannot find type `Ty` in this scope [E0412]
}
