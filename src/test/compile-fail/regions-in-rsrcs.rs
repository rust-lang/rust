struct yes0 {
  x: &uint,
}

impl yes0 : Drop {
    fn finalize() {}
}

struct yes1 {
  x: &self/uint,
}

impl yes1 : Drop {
    fn finalize() {}
}

struct yes2 {
  x: &foo/uint, //~ ERROR named regions other than `self` are not allowed as part of a type declaration
}

impl yes2 : Drop {
    fn finalize() {}
}

fn main() {}
