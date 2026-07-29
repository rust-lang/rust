//@ run-pass

extern "C" fn f() {
}

fn is_fn<F>(f: F) where F: Fn() {
    f();
}

fn main() {
    let _x: extern "C" fn() = f; // OK
    is_fn(f);
}
