// run-rustfix

fn main() {
    let _x = ~1; //~ ERROR cannot be used as a unary operator
}
