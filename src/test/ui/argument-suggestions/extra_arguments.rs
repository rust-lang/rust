fn empty() {}
fn one_arg(_a: i32) {}
fn two_arg_same(_a: i32, _b: i32) {}
fn two_arg_diff(_a: i32, _b: &str) {}

fn main() {
  empty(""); //~ ERROR this function takes

  one_arg(1, 1); //~ ERROR this function takes
  one_arg(1, ""); //~ ERROR this function takes
  one_arg(1, "", 1.0); //~ ERROR this function takes

  two_arg_same(1, 1, 1); //~ ERROR this function takes
  two_arg_same(1, 1, 1.0); //~ ERROR this function takes

  two_arg_diff(1, 1, ""); //~ ERROR this function takes
  two_arg_diff(1, "", ""); //~ ERROR this function takes
  two_arg_diff(1, 1, "", ""); //~ ERROR this function takes
  two_arg_diff(1, "", 1, ""); //~ ERROR this function takes

  // Check with weird spacing and newlines
  two_arg_same(1, 1,     ""); //~ ERROR this function takes
  two_arg_diff(1, 1,     ""); //~ ERROR this function takes
  two_arg_same( //~ ERROR this function takes
    1,
    1,
    ""
  );

  two_arg_diff( //~ ERROR this function takes
    1,
    1,
    ""
  );
}
