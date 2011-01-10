// error-pattern: unresolved name: baz
import zed.bar;
import zed.baz;
mod zed {
  fn bar() {
    log "bar";
  }
}
fn main(vec[str] args) {
   bar();
}
