// Macro from prelude is shadowed by non-existent import recovered as `Def::Err`.

use std::assert; //~ ERROR unresolved import `std::assert`

fn main() {
    assert!(true);
}
