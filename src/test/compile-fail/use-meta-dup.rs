// error-pattern:duplicate meta item `name`

use std(name = "std", name = "nonstd");

fn main() { }
