// error-pattern:fail

fn f() {
    let a = @0;
    fail;
}

fn g() {
    let b = @0;
    f();
}

fn main() {
    let a = @0;
    g();
}