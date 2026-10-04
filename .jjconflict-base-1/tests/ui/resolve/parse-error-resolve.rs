mod parse_error;
use parse_error::Canonical; // ok, `parse_error.rs` had parse errors

fn main() {
    let _ = "" + 1; //~ ERROR E0369
    parse_error::Canonical.foo(); // ok, `parse_error.rs` had parse errors
}

//~? ERROR expected one of `+`, `,`, `::`, `=`, or `>`, found `From`
