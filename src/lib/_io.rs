type buf_reader = unsafe obj {
  fn read() -> vec[u8];
};

fn default_bufsz() -> uint {
  ret 4096u;
}

fn new_buf() -> vec[u8] {
  ret _vec.alloc[u8](default_bufsz());
}

fn new_buf_reader(str s) -> buf_reader {

  unsafe obj fd_buf_reader(int fd, mutable vec[u8] buf) {

    fn read() -> vec[u8] {

      // Ensure our buf is singly-referenced.
      if (_vec.rustrt.refcount[u8](buf) != 1u) {
        buf = new_buf();
      }

      auto len = _vec.len[u8](buf);
      auto vbuf = _vec.buf[u8](buf);
      auto count = os.libc.read(fd, vbuf, len);

      if (count < 0) {
        log "error filling buffer";
        log sys.rustrt.last_os_error();
        fail;
      } else {
        ret buf;
      }
    }

    drop {
      os.libc.close(fd);
    }
  }

  auto fd = os.libc.open(_str.buf(s), 0);
  if (fd < 0) {
    log "error opening file";
    log sys.rustrt.last_os_error();
    fail;
  }
  ret fd_buf_reader(fd, new_buf());
}
