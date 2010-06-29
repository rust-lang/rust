import vbuf = rustrt.vbuf;

native "rust" mod rustrt {
  type vbuf;
  fn vec_buf[T](vec[T] v) -> vbuf;
  fn vec_len[T](vec[T] v) -> uint;
  fn vec_alloc[T](int n_elts) -> vec[T];
}

fn alloc[T](int n_elts) -> vec[T] {
  ret rustrt.vec_alloc[T](n_elts);
}

fn init[T](&T t, int n_elts) -> vec[T] {
  let vec[T] v = alloc[T](n_elts);
  let int i = n_elts;
  while (i > 0) {
    i -= 1;
    v += vec(t);
  }
  ret v;
}

fn len[T](vec[T] v) -> uint {
  ret rustrt.vec_len[T](v);
}

fn buf[T](vec[T] v) -> vbuf {
  ret rustrt.vec_buf[T](v);
}

// Ought to take mutable &vec[T] v and just mutate it instead of copy
// and return.  Blocking on issue #89 for this.
fn grow[T](mutable vec[T] v, int n, T initval) -> vec[T] {
  let int i = n;
  while (i > 0) {
    i -= 1;
    v += vec(initval);
  }
  ret v;
}
