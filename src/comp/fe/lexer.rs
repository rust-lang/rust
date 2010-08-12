import std._io.buf_reader;

iter buffers(buf_reader rdr) -> vec[u8] {
  while (true) {
    let vec[u8] v = rdr.read();
    if (std._vec.len[u8](v) == 0u) {
      ret;
    }
    put v;
  }
}

iter bytes(buf_reader rdr) -> u8 {
  for each (vec[u8] buf in buffers(rdr)) {
    for (u8 b in buf) {
      // FIXME: doesn't compile at the moment.
      // put b;
    }
  }
}

//
// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C ../.. 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
//
