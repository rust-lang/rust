trait bar { fn dup() -> self; fn blah<X>(); }
impl int: bar { fn dup() -> int { self } fn blah<X>() {} }
impl uint: bar { fn dup() -> uint { self } fn blah<X>() {} }

fn main() {
    10.dup::<int>(); //~ ERROR does not take type parameters
    10.blah::<int, int>(); //~ ERROR incorrect number of type parameters
    (10 as bar).dup(); //~ ERROR contains a self-type
}
