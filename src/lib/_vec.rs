import vbuf = rustrt.vbuf;
import op = util.operator;

native "rust" mod rustrt {
  type vbuf;
  fn vec_buf[T](vec[T] v) -> vbuf;
  fn vec_len[T](vec[T] v) -> uint;
  fn vec_alloc[T](uint n_elts) -> vec[T];
}

fn alloc[T](uint n_elts) -> vec[T] {
  ret rustrt.vec_alloc[T](n_elts);
}

type init_op[T] = fn(uint i) -> T;

fn init_fn[T](&init_op[T] op, uint n_elts) -> vec[T] {
  let vec[T] v = alloc[T](n_elts);
  let uint i = n_elts;
  while (i > uint(0)) {
    i -= uint(1);
    v += vec(op(i));
  }
  ret v;
}

fn init_elt[T](&T t, uint n_elts) -> vec[T] {
  // FIXME: should be:
  // fn elt_op[X](X x, uint i) -> X { ret x; }
  // auto inner = bind elt_op[T](t, _);
  // ret init_fn[T](inner, n_elts);
  // but this does not work presently.
  let vec[T] v = alloc[T](n_elts);
  let uint i = n_elts;
  while (i > uint(0)) {
    i -= uint(1);
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

fn map[T,U](&op[T,U] f, &vec[T] v) -> vec[U] {
  // FIXME: should be
  // let vec[U] u = alloc[U](len[T](v));
  // but this does not work presently.
  let vec[U] u = vec();
  for (T ve in v) {
    u += vec(f[T,U](ve));
  }
  ret u;
}

