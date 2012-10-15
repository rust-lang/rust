// xfail-test
fn main() {
    let foo = 100;

    enum Stuff {
        Bar = foo
    }

    log(error, Bar);
}
