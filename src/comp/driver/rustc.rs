// -*- rust -*-

fn main(vec[str] args) -> () {
  let int i = 0;
  for (str filename in args) {
    if (i > 0) {
      auto br = std._io.mk_buf_reader(filename);
      log "opened file: " + filename;
    }
    i += 1;
  }
}
