type buf_reader = obj {
  fn read(vec[u8] buf) -> uint;
};

type buf_writer = obj {
  fn write(vec[u8] buf) -> uint;
};

fn mk_buf_reader(str s) -> buf_reader {

  obj fd_reader(int fd) {
    fn read(vec[u8] v) -> uint {
      auto len = _vec.len[u8](v);
      auto buf = _vec.buf[u8](v);
      auto count = os.libc.read(fd, buf, len);
      if (count < 0) {
        log "error filling buffer";
        log sys.rustrt.last_os_error();
        fail;
      } else {
        ret uint(count);
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
  ret fd_reader(fd);
}
