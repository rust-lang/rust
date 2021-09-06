// check-pass

use std::io::Write;

fn main() -> Result<(), std::io::Error> {
    vec! { 1, 2, 3 }.len();
    write! { vec![], "" }?;
    Ok(())
}
