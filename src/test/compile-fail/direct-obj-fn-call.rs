
// error-pattern: mismatched types

obj x() {
  fn hello() {
    log "hello";
  }
}

fn main() {
  x.hello();
}