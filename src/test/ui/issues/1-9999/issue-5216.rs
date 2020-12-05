fn f() { }
struct S(Box<dyn FnMut()>);
pub static C: S = S(f); //~ ERROR mismatched types


fn g() { }
type T = Box<dyn FnMut()>;
pub static D: T = g; //~ ERROR mismatched types

fn main() {}
