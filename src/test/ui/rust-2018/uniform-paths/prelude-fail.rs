// edition:2018

// Built-in macro
use env as env_imported; //~ ERROR cannot import a built-in macro

// Tool attribute
use rustfmt::skip as imported_rustfmt_skip; //~ ERROR unresolved import `rustfmt`

fn main() {
    env_imported!("PATH");
}
