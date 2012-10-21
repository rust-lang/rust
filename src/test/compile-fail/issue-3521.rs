fn main() {
    let foo = 100;

    enum Stuff {
        Bar = foo //~ ERROR attempt to use a non-constant value in a constant
    }

    log(error, Bar);
}
