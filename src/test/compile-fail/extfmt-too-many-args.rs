// xfail-stage0
// error-pattern:too many arguments

use std;

fn main() { let s = #fmt("%s", "test", "test"); }