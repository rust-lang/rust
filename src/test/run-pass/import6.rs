import bar::baz;
import foo::zed;
mod foo {
  mod zed {
    fn baz() {
      log "baz";
    }
  }
}
mod bar {
  import zed::baz;
}
fn main(vec[str] args) {
  baz();
}
