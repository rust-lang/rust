iface bar { fn dup() -> self; fn blah<X>(); }
impl of bar for int { fn dup() -> int { self } fn blah<X>() {} }
impl of bar for uint { fn dup() -> uint { self } fn blah<X>() {} }
impl of bar for uint { fn dup() -> uint { self } fn blah<X>() {} }

fn main() {
    10.dup::<int>(); //~ ERROR does not take type parameters
    10.blah::<int, int>(); //~ ERROR incorrect number of type parameters
    10u.dup(); //~ ERROR multiple applicable methods
    (10 as bar).dup(); //~ ERROR contains a self type
}
