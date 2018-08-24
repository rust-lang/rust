type x = Vec<x>;
//~^ ERROR cycle detected

fn main() { let b: x = Vec::new(); }
