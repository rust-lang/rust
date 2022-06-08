struct X {}

fn two_args(_a: i32, _b: f32) {}
fn three_args(_a: i32, _b: f32, _c: &str) {}
fn four_args(_a: i32, _b: f32, _c: &str, _d: X) {}

fn main() {
  two_args(1.0, 1);
  //~^ ERROR mismatched types
  //~| ERROR mismatched types
  three_args(1.0,   1,  "");
  //~^ ERROR mismatched types
  //~| ERROR mismatched types
  three_args(  1,  "", 1.0);
  //~^ ERROR mismatched types
  //~| ERROR mismatched types
  three_args( "", 1.0,   1);
  //~^ ERROR mismatched types
  //~| ERROR mismatched types
  four_args(1.0, 1, X {}, "");
  //~^ ERROR mismatched types
  //~| ERROR mismatched types
  //~| ERROR mismatched types
  //~| ERROR mismatched types
}
