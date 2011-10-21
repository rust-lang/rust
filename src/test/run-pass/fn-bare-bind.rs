fn f(i: int) {
    assert i == 10;
}

fn main() {
    // Binding a bare function turns it into a shared closure
    let g: fn@() = bind f(10);
    g();
}