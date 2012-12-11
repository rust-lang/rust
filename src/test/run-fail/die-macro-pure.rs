// error-pattern:test

pure fn f() {
    die!(~"test");
}

fn main() {
    f();
}