// error-pattern:fail
fn f(-_a: @int) {
    fail;
}

fn main() {
    let a = @0;
    f(move a);
}