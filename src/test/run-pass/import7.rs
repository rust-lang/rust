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
  mod foo {
    mod zed {
    }
  }
}
fn main(vec[str] args) {
  baz();
}
