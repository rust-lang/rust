// error-pattern:too many arguments

// xfail-test (issue #936)

use std;

fn main() { let s = #fmt["%s", "test", "test"]; }
