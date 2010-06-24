// error-pattern: unknown module item
mod foo {
  export x;
  fn x(int y) {
    log y;
  }
  fn z(int y) {
    log y;
  }
}

fn main() {
  foo.z(10);
}
