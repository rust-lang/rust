// xfail-test
fn function() -> &[mut int] {
    let mut x: &static/[mut int] = &[mut 1,2,3];
    x[0] = 12345;
    x //~ ERROR bad
}

fn main() {
    let x = function();
    error!("%?", x);
}