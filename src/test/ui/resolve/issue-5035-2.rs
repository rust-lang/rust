trait I {}
type K = I+'static;

fn foo(_x: K) {}
//~^ ERROR the size for values of type

fn main() {}
