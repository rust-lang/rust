// -*- rust -*-

fn main(vec[str] args) -> () {

  log "This is the rust 'self-hosted' compiler.";
  log "The one written in rust.";
  log "It does nothing yet, it's a placeholder.";
  log "You want rustboot, the compiler next door.";

  auto i = 0;
  for (str filename in args) {
    if (i > 0) {
      auto br = std._io.new_buf_reader(filename);
      log "opened file: " + filename;
      for (u8 b in br.read()) {
        log b;
      }
    }
    i += 1;
  }

}
