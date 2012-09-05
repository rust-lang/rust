struct yes0 {
  let x: &uint;
  drop {}
}

struct yes1 {
  let x: &self/uint;
  drop {}
}

struct yes2 {
  let x: &foo/uint; //~ ERROR named regions other than `self` are not allowed as part of a type declaration
  drop {}
}

fn main() {}