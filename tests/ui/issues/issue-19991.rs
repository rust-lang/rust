// Test if the sugared `if let` construct correctly prints "missing an else clause" when an else
// clause does not exist, instead of the unsympathetic "`match` arms have incompatible types"

//@ dont-require-annotations: NOTE

fn main() {
    if let Some(homura) = Some("madoka") { //~  ERROR missing an `else` clause
                                           //~| NOTE expected integer, found `()`
        765
    };
}
