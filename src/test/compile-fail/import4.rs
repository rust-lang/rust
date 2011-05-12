// xfail-boot
// error-pattern: cyclic import

import zed::bar;
import bar::zed;

fn main(vec[str] args) {
  log "loop";
}
