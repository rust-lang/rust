//@ run-rustfix

fn main() {
    let _x = ~1; //~ ERROR cannot be used as a unary operator
    let _y = not 1; //~ ERROR unexpected `1` after identifier
    let _z = not false; //~ ERROR unexpected keyword `false` after identifier
    let _a = not true; //~ ERROR unexpected keyword `true` after identifier
    let v = 1 + 2;
    let _v = not v; //~ ERROR unexpected `v` after identifier
}
