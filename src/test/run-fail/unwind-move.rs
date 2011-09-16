// error-pattern:fail
fn f(-a: @int) {
    fail;
}

fn main() {
    let a = @0;
    f(a);
}