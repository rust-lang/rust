import rustrt.sbuf;

native "rust" mod rustrt {
  type sbuf;
  fn str_buf(str s) -> sbuf;
  fn str_len(str s) -> uint;
  fn str_alloc(uint n_bytes) -> str;
  fn refcount[T](str s) -> uint;
}

fn is_utf8(vec[u8] v) -> bool {
  fail; // FIXME
}

fn is_ascii(str s) -> bool {
  let uint i = len(s);
  while (i > 0u) {
    i -= 1u;
    if ((s.(i) & 0x80u8) != 0u8) {
      ret false;
    }
  }
  ret true;
}

fn alloc(uint n_bytes) -> str {
  ret rustrt.str_alloc(n_bytes);
}

fn len(str s) -> uint {
  ret rustrt.str_len(s);
}

fn buf(str s) -> sbuf {
  ret rustrt.str_buf(s);
}

fn bytes(&str s) -> vec[u8] {
  fn ith(str s, uint i) -> u8 {
    ret s.(i);
  }
  ret _vec.init_fn[u8](bind ith(s, _), _str.len(s));
}
