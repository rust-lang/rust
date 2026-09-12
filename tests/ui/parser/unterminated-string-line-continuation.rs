//! A backslash at the end of a line continues a string literal onto the next line on purpose, so
//! such a literal must not be blamed for a later unterminated string.
//!
//! See <https://github.com/rust-lang/rust/issues/97001>.

fn main() {
    let _ = "this one is \
             deliberately split";
    let _ = "unterminated;
}
//~^^ ERROR unterminated double quote string
