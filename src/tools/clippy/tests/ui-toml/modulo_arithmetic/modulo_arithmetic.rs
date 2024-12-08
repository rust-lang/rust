#![warn(clippy::modulo_arithmetic)]

fn main() {
    let a = -1;
    let b = 2;
    let c = a % b == 0;
    let c = a % b != 0;
    let c = 0 == a % b;
    let c = 0 != a % b;
}
