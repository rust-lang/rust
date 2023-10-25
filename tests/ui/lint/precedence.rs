// run-rustfix
// check-pass

fn main() {
    let _ = 1 << 2 + 3;
    //~^ WARN operator precedence can trip the unwary
    let _ = 1 + 2 << 3;
    //~^ WARN operator precedence can trip the unwary
    let _ = 4 >> 1 + 1;
    //~^ WARN operator precedence can trip the unwary
    let _ = 1 + 3 >> 2;
    //~^ WARN operator precedence can trip the unwary
    let _ = 1 ^ 1 - 1;
    //~^ WARN operator precedence can trip the unwary
    let _ = 3 | 2 - 1;
    //~^ WARN operator precedence can trip the unwary
    let _ = 3 & 5 - 2;
    //~^ WARN operator precedence can trip the unwary
    let _ = 1 + 2 << 3 + 1;
    //~^ WARN operator precedence can trip the unwary
    let _ = -1i32.abs();
    //~^ WARN unary minus has lower precedence than method call
    let _ = -1f32.abs();
    //~^ WARN unary minus has lower precedence than method call
    let _ = -1f64.asin();
    //~^ WARN unary minus has lower precedence than method call
    let _ = -1f64.asinh();
    //~^ WARN unary minus has lower precedence than method call
    let _ = -1f64.tan();
    //~^ WARN unary minus has lower precedence than method call
    let _ = -1f64.tanh();
    //~^ WARN unary minus has lower precedence than method call
    let _ = -1.0_f64.cos().cos();
    //~^ WARN unary minus has lower precedence than method call
    let _ = -1.0_f64.cos().sin();
    //~^ WARN unary minus has lower precedence than method call
    let _ = -1.0_f64.sin().cos();
    //~^ WARN unary minus has lower precedence than method call
    let _ = -1f64.sin().sin();
    //~^ WARN unary minus has lower precedence than method call

    // These should not trigger an error
    let _ = (-1i32).abs();
    let _ = (-1f32).abs();
    let _ = -(1i32).abs();
    let _ = -(1f32).abs();
    let _ = -(1i32.abs());
    let _ = -(1f32.abs());
}
