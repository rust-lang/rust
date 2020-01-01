// compile-flags: -Z ui-testing=no

// Line number < 10
type A = B; //~ ERROR

// http://rust-lang.org/COPYRIGHT.
//

// Line number >=10, <100
type C = D; //~ ERROR



















































































// Line num >=100
type E = F; //~ ERROR

fn main() {}
