// More nuanced test cases for invalid arguments #65853

struct X {}

fn one_arg(_a: i32) {}
fn two_arg_same(_a: i32, _b: i32) {}
fn two_arg_diff(_a: i32, _b: f32) {}
fn three_arg_diff(_a: i32, _b: f32, _c: &str) {}
fn three_arg_repeat(_a: i32, _b: i32, _c: &str) {}

fn main() {
  // Providing an incorrect argument for a single parameter function
  one_arg(1.0); //~ ERROR mismatched types

  // Providing one or two invalid arguments to a two parameter function
  two_arg_same(1, ""); //~ ERROR mismatched types
  two_arg_same("", 1); //~ ERROR mismatched types
  two_arg_same("", ""); //~ ERROR arguments to this function are incorrect
  two_arg_diff(1, ""); //~ ERROR mismatched types
  two_arg_diff("", 1.0); //~ ERROR mismatched types
  two_arg_diff("", ""); //~ ERROR arguments to this function are incorrect

  // Providing invalid arguments to a three parameter function
  three_arg_diff(X{}, 1.0, ""); //~ ERROR mismatched types
  three_arg_diff(1, X {}, ""); //~ ERROR mismatched types
  three_arg_diff(1, 1.0, X {}); //~ ERROR mismatched types

  three_arg_diff(X {}, X {}, ""); //~ ERROR arguments to this function are incorrect
  three_arg_diff(X {}, 1.0, X {}); //~ ERROR arguments to this function are incorrect
  three_arg_diff(1, X {}, X {}); //~ ERROR arguments to this function are incorrect

  three_arg_diff(X {}, X {}, X {}); //~ ERROR arguments to this function are incorrect

  three_arg_repeat(X {}, 1, ""); //~ ERROR mismatched types
  three_arg_repeat(1, X {}, ""); //~ ERROR mismatched types
  three_arg_repeat(1, 1, X {}); //~ ERROR mismatched types

  three_arg_repeat(X {}, X {}, ""); //~ ERROR arguments to this function are incorrect
  three_arg_repeat(X {}, 1, X {}); //~ ERROR arguments to this function are incorrect
  three_arg_repeat(1, X {}, X{}); //~ ERROR arguments to this function are incorrect

  three_arg_repeat(X {}, X {}, X {}); //~ ERROR arguments to this function are incorrect
}
