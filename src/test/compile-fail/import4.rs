// error-pattern: recursive import

import zed.bar;
import bar.zed;

fn main(vec[str] args) {
  log "loop";
}
