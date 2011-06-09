// error-pattern: parameters were supplied

fn f(int x) {
}

fn main() {
  let () i;
  i = f();
}
