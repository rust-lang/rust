// Test if the sugared if-let construct correctly prints "missing an else clause" when an else
// clause does not exist, instead of the unsympathetic "match arms have incompatible types"

fn main() {
    if let Some(homura) = Some("madoka") { //~  ERROR missing an else clause
                                           //~| expected type `()`
                                           //~| found type `{integer}`
                                           //~| expected (), found integer
        765
    };
}
