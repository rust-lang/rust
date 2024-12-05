mod parse_error;
use parse_error::Canonical; //~ ERROR E0432

fn main() {
    let _ = "" + 1; //~ ERROR E0369
    parse_error::Canonical.foo(); // ok, `parse_error.rs` had parse errors
}
