fn main() {
    let foo = 100;

    const y: int = foo + 1; //~ ERROR: attempt to use a non-constant value in a constant

    log(error, y);
}
