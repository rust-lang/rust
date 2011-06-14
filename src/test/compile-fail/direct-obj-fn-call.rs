// error-pattern: base type for expr_field

obj x() {
  fn hello() {
    log "hello";
  }
}

fn main() {
  x.hello();
}