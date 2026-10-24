const A: i32 = B; //~ ERROR cycle detected
//@ ignore-parallel-frontend query cycle
const B: i32 = A;

fn main() { }
