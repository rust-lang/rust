const A: i32 = B; //~ ERROR cycle detected

const B: i32 = A;

fn main() { }
