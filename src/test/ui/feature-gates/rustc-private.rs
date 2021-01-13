// gate-test-rustc_private

extern crate libc; //~ ERROR  use of unstable library feature 'rustc_private'

fn main() {}
