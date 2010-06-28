
// error-pattern: is not a mod

obj x() {
  fn hello() {
    log "hello";
  }
}

fn main() {
  x.hello();
}