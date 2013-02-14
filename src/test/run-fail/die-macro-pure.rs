// error-pattern:test

pure fn f() {
    fail!(~"test");
}

fn main() {
    f();
}
