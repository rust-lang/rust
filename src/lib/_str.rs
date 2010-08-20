import rustrt.sbuf;

import std._vec.rustrt.vbuf;

native "rust" mod rustrt {
  type sbuf;
  fn str_buf(str s) -> sbuf;
  fn str_byte_len(str s) -> uint;
  fn str_alloc(uint n_bytes) -> str;
  fn str_from_vec(vec[u8] b) -> str;
  fn refcount[T](str s) -> uint;
}

fn eq(str a, str b) -> bool {
  let uint i = byte_len(a);
  if (byte_len(b) != i) {
    ret false;
  }
  while (i > 0u) {
    i -= 1u;
    auto cha = a.(i);
    auto chb = b.(i);
    if (cha != chb) {
      ret false;
    }
  }
  ret true;
}

fn is_utf8(vec[u8] v) -> bool {
  fail; // FIXME
}

fn is_ascii(str s) -> bool {
  let uint i = byte_len(s);
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

// Returns the number of bytes (a.k.a. UTF-8 code units) in s.
// Contrast with a function that would return the number of code
// points (char's), combining character sequences, words, etc.  See
// http://icu-project.org/apiref/icu4c/classBreakIterator.html for a
// way to implement those.
fn byte_len(str s) -> uint {
  ret rustrt.str_byte_len(s);
}

fn buf(str s) -> sbuf {
  ret rustrt.str_buf(s);
}

fn bytes(str s) -> vec[u8] {
  /* FIXME (issue #58):
   * Should be...
   *
   *  fn ith(str s, uint i) -> u8 {
   *      ret s.(i);
   *  }
   *  ret _vec.init_fn[u8](bind ith(s, _), byte_len(s));
   *
   * but we do not correctly decrement refcount of s when
   * the binding dies, so we have to do this manually.
   */
  let uint n = _str.byte_len(s);
  let vec[u8] v = _vec.alloc[u8](n);
  let uint i = 0u;
  while (i < n) {
    v += vec(s.(i));
    i += 1u;
  }
  ret v;
}

fn from_bytes(vec[u8] v) : is_utf8(v) -> str {
  ret rustrt.str_from_vec(v);
}

fn refcount(str s) -> uint {
  // -1 because calling this function incremented the refcount.
  ret rustrt.refcount[u8](s) - 1u;
}
