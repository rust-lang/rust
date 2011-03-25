// xfail-stage0

// error-pattern: malformed name

fn main() {
  let x.y[int].z foo;
}
