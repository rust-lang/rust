#![warn(clippy::modulo_arithmetic)]

fn main() {
    let a = -1;
    let b = 2;
    let c = a % b == 0;
    //~^ modulo_arithmetic
    let c = a % b != 0;
    //~^ modulo_arithmetic
    let c = 0 == a % b;
    //~^ modulo_arithmetic
    let c = 0 != a % b;
    //~^ modulo_arithmetic
}
