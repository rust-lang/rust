fn main() {
    let foo = 100;
    const quux: int = 5;

    enum Stuff {
        Bar = quux
    }

    assert (Bar as int == quux);
}
