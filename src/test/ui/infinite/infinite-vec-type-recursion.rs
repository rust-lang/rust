type X = Vec<X>;
//~^ ERROR cycle detected

fn main() { let b: X = Vec::new(); }
