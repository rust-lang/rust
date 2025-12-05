//@ edition:2015
use main::bar; //~ ERROR unresolved import `main`

fn main() { println!("foo"); }
