// Cases where multiple argument suggestions are mixed

struct X {}

fn two_args(_a: i32, _b: f32) {}
fn three_args(_a: i32, _b: f32, _c: &str) {}

fn main() {
  // Extra + Invalid
  two_args(1, "", X {}); //~ ERROR function takes
  three_args(1, "", X {}, ""); //~ ERROR function takes

  // Missing and Invalid
  three_args(1, X {}); //~ ERROR function takes

  // Missing and Extra
  three_args(1, "", X {}); //~ ERROR arguments to this function are incorrect

  // Swapped and Invalid
  three_args("", X {}, 1); //~ ERROR arguments to this function are incorrect

  // Swapped and missing
  three_args("", 1); //~ ERROR function takes
}
