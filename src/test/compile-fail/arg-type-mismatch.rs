
// error-pattern: mismatched types

fn f(int x) {
}

fn main() {
  let () i;
  i = f(());
}