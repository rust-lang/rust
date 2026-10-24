#![crate_name="0"]

fn main() {}

//~vv ERROR mismatched closing delimiter: `]`
//~v ERROR this file contains an unclosed delimiter
fn r()->i{0|{#[cfg(r(0{]0
