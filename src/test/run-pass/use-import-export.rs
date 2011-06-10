mod foo {
  export x;
  use std (vers="0.1");
  fn x() -> int { ret 1; }
}

mod bar {
  use std (vers="0.1");
  export y;
  fn y() -> int { ret 1; }
}

fn main() {
  foo::x();
  bar::y();
}

