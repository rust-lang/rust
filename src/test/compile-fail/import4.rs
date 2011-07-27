// error-pattern: cyclic import

import zed::bar;
import bar::zed;

fn main(args: vec[str]) { log "loop"; }