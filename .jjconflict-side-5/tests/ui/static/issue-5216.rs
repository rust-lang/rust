fn f() { }
struct S(Box<dyn FnMut() + Sync>);
pub static C: S = S(f); //~ ERROR mismatched types


fn g() { }
type T = Box<dyn FnMut() + Sync>;
pub static D: T = g; //~ ERROR mismatched types

fn main() {}
