// error-pattern:expected `fn()` but found `*u8`
crust fn f() {
}

fn main() {
    // Crust functions are *u8 types
    let _x: fn() = f;
}