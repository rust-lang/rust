fn f<T>(i: T, j: T, k: T) {
    assert i == j;
    assert j != k;
}

fn main() {
    // Binding a bare function turns it into a shared closure
    let g: fn@() = bind f(10, 10, 20);
    g();
}