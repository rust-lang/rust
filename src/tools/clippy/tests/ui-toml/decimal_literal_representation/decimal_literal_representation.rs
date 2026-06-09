#![warn(clippy::decimal_literal_representation)]
fn main() {
    let _ = 8388608;
    let _ = 16777215;
    //~^ ERROR: integer literal has a better hexadecimal representation
}
