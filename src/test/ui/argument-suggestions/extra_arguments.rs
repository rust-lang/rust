fn empty() {}
fn one_arg(_a: i32) {}
fn two_arg_same(_a: i32, _b: i32) {}
fn two_arg_diff(_a: i32, _b: &str) {}

fn main() {
  empty(""); //~ ERROR arguments to this function are incorrect
  
  one_arg(1, 1); //~ ERROR arguments to this function are incorrect
  one_arg(1, ""); //~ ERROR arguments to this function are incorrect
  one_arg(1, "", 1.0); //~ ERROR arguments to this function are incorrect

  two_arg_same(1, 1, 1); //~ ERROR arguments to this function are incorrect
  two_arg_same(1, 1, 1.0); //~ ERROR arguments to this function are incorrect

  two_arg_diff(1, 1, ""); //~ ERROR arguments to this function are incorrect
  two_arg_diff(1, "", ""); //~ ERROR arguments to this function are incorrect
  two_arg_diff(1, 1, "", ""); //~ ERROR arguments to this function are incorrect
  two_arg_diff(1, "", 1, ""); //~ ERROR arguments to this function are incorrect
}