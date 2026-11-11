trait I {}
type K = dyn I + 'static;

fn foo(_x: K) {}
//~^ ERROR the size for values of type

fn main() {}
