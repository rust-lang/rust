//@ edition:2018

#[bench] //~ ERROR use of unstable library feature `test`
fn bench() {}

use bench as _; //~ ERROR use of unstable library feature `test`
fn main() {}
