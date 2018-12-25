fn main() {
    let x: bool;
    while x { } //~ ERROR use of possibly uninitialized variable: `x`
}
