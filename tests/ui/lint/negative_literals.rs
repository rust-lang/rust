//@ check-fail

fn main() {
    let _ = -1i32.abs();
    //~^ ERROR `-` has lower precedence than method calls
    let _ = -1f32.abs();
    //~^ ERROR `-` has lower precedence than method calls
    let _ = -1f64.asin();
    //~^ ERROR `-` has lower precedence than method calls
    let _ = -1f64.asinh();
    //~^ ERROR `-` has lower precedence than method calls
    let _ = -1f64.tan();
    //~^ ERROR `-` has lower precedence than method calls
    let _ = -1f64.tanh();
    //~^ ERROR `-` has lower precedence than method calls
    let _ = -1.0_f64.cos().cos();
    //~^ ERROR `-` has lower precedence than method calls
    let _ = -1.0_f64.cos().sin();
    //~^ ERROR `-` has lower precedence than method calls
    let _ = -1.0_f64.sin().cos();
    //~^ ERROR `-` has lower precedence than method calls
    let _ = -1f64.sin().sin();
    //~^ ERROR `-` has lower precedence than method calls

    dbg!( -1.0_f32.cos() );
    //~^ ERROR `-` has lower precedence than method calls

    // should not warn
    let _ = (-1i32).abs();
    let _ = (-1f32).abs();
    let _ = -(1i32).abs();
    let _ = -(1f32).abs();
    let _ = -(1i32.abs());
    let _ = -(1f32.abs());
}
