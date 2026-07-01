//@ edition:2018

// Tool
use rustfmt as imported_rustfmt; //~ ERROR unresolved import `rustfmt`

// Tool attribute
use rustfmt::skip as imported_rustfmt_skip; //~ ERROR unresolved import `rustfmt`

fn main() {}
