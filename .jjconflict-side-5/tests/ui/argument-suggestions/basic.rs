// Some basic "obvious" cases for the heuristic error messages added for #65853
// One for each of the detected cases

enum E { X, Y }
enum F { X2, Y2 }
struct G {}
struct H {}
struct X {}
struct Y {}
struct Z {}


fn invalid(_i: u32) {}
fn extra() {}
fn missing(_i: u32) {}
fn swapped(_i: u32, _s: &str) {}
fn permuted(_x: X, _y: Y, _z: Z) {}

fn main() {
    invalid(1.0); //~ ERROR mismatched types
    extra(""); //~ ERROR function takes
    missing(); //~ ERROR function takes
    swapped("", 1); //~ ERROR arguments to this function are incorrect
    permuted(Y {}, Z {}, X {}); //~ ERROR arguments to this function are incorrect

    let closure = |x| x;
    closure(); //~ ERROR function takes
}
