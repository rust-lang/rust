//@ edition:2018

// Tool attribute
use rustfmt::skip as imported_rustfmt_skip; //~ ERROR unresolved import `rustfmt`

fn main() {}
