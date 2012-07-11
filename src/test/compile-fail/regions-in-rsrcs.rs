class yes0 {
  let x: &uint;
  new(x: &uint) { self.x = x; }
  drop {}
}

class yes1 {
  let x: &self.uint;
  new(x: &self.uint) { self.x = x; }
  drop {}
}

class yes2 {
  let x: &foo.uint; //~ ERROR named regions other than `self` are not allowed as part of a type declaration
  new(x: &foo.uint) { self.x = x; } //~ ERROR named regions other than `self` are not allowed as part of a type declaration
  drop {}
}

fn main() {}