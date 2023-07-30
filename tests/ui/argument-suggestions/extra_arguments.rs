fn empty() {}
fn one_arg<T>(_a: T) {}
fn two_arg_same(_a: i32, _b: i32) {}
fn two_arg_diff(_a: i32, _b: &str) {}

macro_rules! foo {
    ($x:expr, ~) => {
        empty($x, 1); //~ ERROR function takes
    };
    ($x:expr, $y:expr) => {
        empty($x, $y); //~ ERROR function takes
    };
    (~, $y:expr) => {
        empty(1, $y); //~ ERROR function takes
    };
}

fn main() {
  empty(""); //~ ERROR function takes
  empty(1, 1); //~ ERROR function takes

  one_arg(1, 1); //~ ERROR function takes
  one_arg(1, ""); //~ ERROR function takes
  one_arg(1, "", 1.0); //~ ERROR function takes

  two_arg_same(1, 1, 1); //~ ERROR function takes
  two_arg_same(1, 1, 1.0); //~ ERROR function takes

  two_arg_diff(1, 1, ""); //~ ERROR function takes
  two_arg_diff(1, "", ""); //~ ERROR function takes
  two_arg_diff(1, 1, "", ""); //~ ERROR function takes
  two_arg_diff(1, "", 1, ""); //~ ERROR function takes

  // Check with weird spacing and newlines
  two_arg_same(1, 1,     ""); //~ ERROR function takes
  two_arg_diff(1, 1,     ""); //~ ERROR function takes
  two_arg_same( //~ ERROR function takes
    1,
    1,
    ""
  );

  two_arg_diff( //~ ERROR function takes
    1,
    1,
    ""
  );

  // Check with macro expansions
  foo!(1, ~);
  foo!(~, 1);
  foo!(1, 1);
  one_arg(1, panic!()); //~ ERROR function takes
  one_arg(panic!(), 1); //~ ERROR function takes
  one_arg(stringify!($e), 1); //~ ERROR function takes

  // Not a macro, but this also has multiple spans with equal source code,
  // but different expansion contexts.
  // https://github.com/rust-lang/rust/issues/114255
  one_arg(for _ in 1.. {}, 1); //~ ERROR function takes
}
