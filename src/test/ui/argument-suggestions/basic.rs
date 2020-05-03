enum E {
    X,
    Y
}
enum F {
    X2,
    Y2
}
struct G {
}
struct H {    
}
struct X {}
struct Y {}
struct Z {}


fn invalid(_i: u32) {}

fn extra() {}

fn missing(_i: u32) {}

fn swapped(_i: u32, _s: &str) {}

fn permuted(_x: X, _y: Y, _z: Z) {}

fn complex(_i: u32, _s: &str, _e: E, _f: F, _g: G, _x: X, _y: Y, _z: Z ) {}

fn main() {
    invalid(1.0); //~ ERROR arguments to this function are incorrect
    extra(&""); //~ ERROR arguments to this function are incorrect
    missing(); //~ ERROR arguments to this function are incorrect
    swapped(&"", 1); //~ ERROR arguments to this function are incorrect
    permuted(Y {}, Z {}, X {}); //~ ERROR arguments to this function are incorrect
    complex(1.0, H {}, &"", G{}, F::X2, Z {}, X {}, Y {}); //~ ERROR arguments to this function are incorrect
}