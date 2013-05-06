// error-pattern:borrowed

fn f(x: &int, y: @mut int) {
    unsafe {
        *y = 2;
    }
}

fn main() {
    let x = @mut 3;
    f(x, x);
}
