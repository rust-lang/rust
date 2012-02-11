crust fn f() {
}

crust fn g() {
}

fn main() {
    // Crust functions are *u8 types
    let a: *u8 = f;
    let b: *u8 = f;
    let c: *u8 = g;

    assert a == b;
    assert a != c;
}
