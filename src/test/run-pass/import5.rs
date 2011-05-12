import foo::bar;
mod foo {
  import zed::bar;
  mod zed {
    fn bar() {
      log "foo";
    }
  }
}

fn main(vec[str] args) {
  bar();
}
