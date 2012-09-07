struct yes0 {
  x: &uint,
  drop {}
}

struct yes1 {
  x: &self/uint,
  drop {}
}

struct yes2 {
  x: &foo/uint, //~ ERROR named regions other than `self` are not allowed as part of a type declaration
  drop {}
}

fn main() {}