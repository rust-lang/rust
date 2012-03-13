// error-pattern:non-scalar cast

fn main() {
  log(debug, { x: 1 } as int);
}
