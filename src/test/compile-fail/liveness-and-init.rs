fn main() {
    let i: int;

    log(debug, false && { i = 5; true });
    log(debug, i); //~ ERROR use of possibly uninitialized variable: `i`
}
