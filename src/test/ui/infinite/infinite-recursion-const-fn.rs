//https://github.com/rust-lang/rust/issues/31364

const fn a() -> usize { b() } //~ ERROR cycle detected when const-evaluating `a` [E0391]
const fn b() -> usize { a() }
const ARR: [i32; a()] = [5; 6];

fn main(){}
