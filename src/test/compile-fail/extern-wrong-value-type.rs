// error-pattern:expected `fn()` but found `*u8`
extern fn f() {
}

fn main() {
    // extern functions are *u8 types
    let _x: fn() = f;
}