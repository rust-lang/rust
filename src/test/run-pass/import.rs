mod foo {
  fn x(int y) {
    log y;
  }
}

mod bar {
  import foo::x;
  import z = foo::x;
  fn thing() {
    x(10);
    z(10);
  }
}

fn main() { bar::thing(); }