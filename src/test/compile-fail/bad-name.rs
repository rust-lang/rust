// xfail-stage0
// xfail-stage1
// xfail-stage2

// error-pattern: malformed name

fn main() {
  let x.y[int].z foo;
}
