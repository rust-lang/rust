// Macro from prelude is shadowed by non-existent import recovered as `Res::Err`.

mod m {}
use m::assert; //~ ERROR unresolved import `m::assert`

fn main() {
    assert!(true);
}
