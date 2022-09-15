// run-rustfix

fn main() {
    let _x = ~1; //~ ERROR cannot be used as a unary operator
    let _y = not 1; //~ ERROR unexpected `1` after identifier
}
