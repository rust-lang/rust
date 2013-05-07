// error-pattern:borrowed

// Test that write guards trigger when mut box is frozen
// as part of argument coercion.

fn f(_x: &int, y: @mut int) {
    *y = 2;
}

fn main() {
    let x = @mut 3;
    f(x, x);
}
