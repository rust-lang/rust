// error-pattern:fail

fn f() {
    fail;
}

fn main() {
    f();
    let a = @0;
}