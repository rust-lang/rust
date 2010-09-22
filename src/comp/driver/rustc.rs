// -*- rust -*-

import fe.parser;
import fe.token;
import me.trans;

fn main(vec[str] args) {

  log "This is the rust 'self-hosted' compiler.";
  log "The one written in rust.";
  log "It does nothing yet, it's a placeholder.";
  log "You want rustboot, the compiler next door.";

  auto i = 0;
  auto sess = session.session();
  for (str filename in args) {
      if (i > 0) {
          auto p = parser.new_parser(sess, filename);
          log "opened file: " + filename;
          auto crate = parser.parse_crate(p);
          trans.translate_crate(sess, crate);
      }
      i += 1;
  }
}


// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C ../.. 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
