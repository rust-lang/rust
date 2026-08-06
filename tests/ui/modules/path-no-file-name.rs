//@ revisions: u w
//@[u] only-unix
//@[w] only-windows

#[path = "."]
mod m; //~ ERROR

fn main() {}
