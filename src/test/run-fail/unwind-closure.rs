// error-pattern:fail

fn f(a: @int) {
    fail;
}

fn main() {
    let g = bind f(@0);
    g();
}