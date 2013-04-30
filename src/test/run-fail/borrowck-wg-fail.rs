// error-pattern:borrowed

fn f(_x: &int, y: @mut int) {
    *y = 2;
}

fn main() {
    let x = @mut 3;
    f(x, x);
}

