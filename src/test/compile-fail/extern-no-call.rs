// error-pattern:expected function or foreign function but found `*u8`
extern fn f() {
}

fn main() {
    f();
}