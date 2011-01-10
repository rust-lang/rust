// error-pattern: unresolved name: zed
import baz.zed.bar;
mod baz {
}
mod zed {
  fn bar() {
    log "bar3";
  }
}
fn main(vec[str] args) {
  bar();
}
